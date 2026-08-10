/*
 * AudioMass — AI Stem Separation (HT-Demucs ONNX, in-browser)
 * Self-contained feature module: downloads the model once from Hugging Face,
 * caches it via the Cache API, runs chunked overlap-add inference with
 * onnxruntime-web (WASM), and writes the chosen stem(s) back to the editor
 * (single stem, with undo) or exports them as WAV files (multi stem).
 *
 * Model: StemSplitio/htdemucs-onnx (MIT) — single-file 4-stem HT-Demucs.
 */
(function ( w, d, PKAE ) {
	'use strict';

	var app = PKAE;

	var MODEL_URL = 'https://huggingface.co/StemSplitio/htdemucs-onnx/resolve/main/htdemucs_fp16weights.onnx';
	var ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js';
	var ORT_WASM_PATH = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';

	var SAMPLE_RATE = 44100;
	var N_SAMPLES = 343980;              // 7.8s @ 44.1kHz — hard-bound to the ONNX graph
	var OVERLAP = Math.floor(N_SAMPLES / 4);
	var STRIDE = N_SAMPLES - OVERLAP;

	var STEM_LIST = [
		{ id: 'vocals',        label: 'Vocals' },
		{ id: 'drums',         label: 'Drums' },
		{ id: 'bass',          label: 'Bass' },
		{ id: 'other',         label: 'Other instruments' },
		{ id: 'accompaniment', label: 'Accompaniment (no vocals)' }
	];
	var STEM_ROW = { drums: 0, bass: 1, other: 2, vocals: 3 };

	var ort_ready = false;
	var ort_loading = false;
	var session = null;
	var model_bytes = null;
	var cancelled = false;

	function engine () { return app.engine; }
	function host () { return app.engine && app.engine.FXPreviewHost; }

	// ---------- runtime / model loading ----------

	function loadOrt () {
		if (ort_ready) return Promise.resolve();
		if (ort_loading) {
			return new Promise(function (ok, no) {
				var t = setInterval(function () {
					if (ort_ready) { clearInterval(t); ok(); }
				}, 200);
			});
		}
		ort_loading = true;
		return new Promise(function (ok, no) {
			var s = d.createElement('script');
			s.src = ORT_URL;
			s.onload = function () { ort_ready = true; ok(); };
			s.onerror = function () { no(new Error('Could not load onnxruntime-web from CDN')); };
			d.head.appendChild(s);
		});
	}

	function fetchModelBytes ( onProgress ) {
		if (model_bytes) return Promise.resolve(model_bytes);

		return w.caches.open('audiomass-stemai-v1').then(function (cache) {
			return cache.match(MODEL_URL).then(function (hit) {
				if (hit) {
					return hit.arrayBuffer().then(function (buf) {
						model_bytes = buf;
						return buf;
					});
				}

				return w.fetch(MODEL_URL).then(function (res) {
					if (!res.ok) throw new Error('Model download failed (HTTP ' + res.status + ')');
					var total = +res.headers.get('content-length') || 174000000;
					var reader = res.body.getReader();
					var parts = [];
					var received = 0;

					function pump () {
						return reader.read().then(function (r) {
							if (r.done) {
								var buf = new ArrayBuffer(received);
								var view = new Uint8Array(buf);
								var off = 0;
								for (var i = 0; i < parts.length; ++i) {
									view.set(parts[i], off);
									off += parts[i].length;
								}
								parts = null;
								// best-effort cache; may fail on quota
								try { cache.put(MODEL_URL, new Response(buf.slice(0))); } catch (e) {}
								model_bytes = buf;
								return buf;
							}
							parts.push(r.value);
							received += r.value.length;
							onProgress && onProgress(received / total, received);
							return pump();
						});
					}
					return pump();
				});
			});
		});
	}

	function getSession ( onProgress ) {
		if (session) return Promise.resolve(session);

		return loadOrt().then(function () {
			onProgress && onProgress('download');
			return fetchModelBytes(function (frac, received) {
				onProgress && onProgress('download', frac, received);
			});
		}).then(function (bytes) {
			onProgress && onProgress('init');
			var ort = w.ort;
			ort.env.wasm.wasmPaths = ORT_WASM_PATH;
			ort.env.wasm.numThreads = w.crossOriginIsolated ?
				Math.min(4, navigator.hardwareConcurrency || 2) : 1;
			return ort.InferenceSession.create(new Uint8Array(bytes), {
				executionProviders: ['wasm'],
				graphOptimizationLevel: 'all'
			});
		}).then(function (sess) {
			session = sess;
			return sess;
		});
	}

	// ---------- audio helpers ----------

	function extractSegment ( buffer, start, dur ) {
		var sr = buffer.sampleRate;
		var off = Math.max(0, Math.min(buffer.length - 1, (start * sr) >> 0));
		var len = Math.max(1, Math.min(buffer.length - off, Math.round(dur * sr)));
		var chans = buffer.numberOfChannels;
		var ac = engine().wavesurfer.backend.ac;
		var seg = ac.createBuffer(chans, len, sr);
		for (var c = 0; c < chans; ++c)
			seg.getChannelData(c).set(buffer.getChannelData(c).subarray(off, off + len));
		return seg;
	}

	function resampleTo44kStereo ( buffer ) {
		function chans ( buf ) {
			return [
				buf.getChannelData(0),
				buf.numberOfChannels > 1 ? buf.getChannelData(1) : buf.getChannelData(0)
			];
		}
		if (buffer.sampleRate === SAMPLE_RATE) return Promise.resolve(chans(buffer));

		var frames = Math.max(1, Math.ceil(buffer.duration * SAMPLE_RATE));
		var oc = new (w.OfflineAudioContext || w.webkitOfflineAudioContext)(2, frames, SAMPLE_RATE);
		var src = oc.createBufferSource();
		src.buffer = buffer;
		src.connect(oc.destination);
		src.start();
		return oc.startRendering().then(chans);
	}

	function stemToBuffer ( stem, targetRate, outChans ) {
		var ac = engine().wavesurfer.backend.ac;
		var srcBuf = ac.createBuffer(2, stem[0].length, SAMPLE_RATE);
		srcBuf.getChannelData(0).set(stem[0]);
		srcBuf.getChannelData(1).set(stem[1]);
		if (targetRate === SAMPLE_RATE && outChans === 2) return Promise.resolve(srcBuf);

		var frames = Math.max(1, Math.round(stem[0].length * targetRate / SAMPLE_RATE));
		var oc = new (w.OfflineAudioContext || w.webkitOfflineAudioContext)(outChans, frames, targetRate);
		var src = oc.createBufferSource();
		src.buffer = srcBuf;
		src.connect(oc.destination);
		src.start();
		return oc.startRendering();
	}

	function makeTransitionWindow ( segment, overlap ) {
		var win = new Float32Array(segment);
		for (var i = 0; i < segment; ++i) win[i] = 1;
		for (var j = 0; j < overlap; ++j) {
			var v = j / overlap;
			win[j] = v;
			win[segment - 1 - j] = v;
		}
		return win;
	}

	// ---------- inference ----------

	// Separates the given model rows in ONE pass — the model predicts all 4
	// stems per run, so multi-stem extraction costs no extra inference time.
	function separateRows ( mix, rows, onProgress ) {
		var ort = w.ort;
		var totalLen = mix[0].length;
		var nChunks = Math.max(1, Math.ceil(totalLen / STRIDE));
		var outs = {};
		for (var ri = 0; ri < rows.length; ++ri)
			outs[rows[ri]] = [new Float32Array(totalLen), new Float32Array(totalLen)];
		var weight = new Float32Array(totalLen);
		var win = makeTransitionWindow(N_SAMPLES, OVERLAP);
		var chunkBuf = new Float32Array(2 * N_SAMPLES);

		var i = 0;
		function step () {
			if (cancelled) return Promise.reject(new Error('Cancelled'));
			if (i >= nChunks) {
				for (var rj = 0; rj < rows.length; ++rj) {
					var o = outs[rows[rj]];
					for (var c = 0; c < 2; ++c)
						for (var s = 0; s < totalLen; ++s)
							o[c][s] /= Math.max(weight[s], 1e-8);
				}
				return Promise.resolve(outs);
			}

			var start = i * STRIDE;
			var end = Math.min(start + N_SAMPLES, totalLen);
			var chunkLen = end - start;
			chunkBuf.fill(0);
			for (var ch = 0; ch < 2; ++ch)
				chunkBuf.subarray(ch * N_SAMPLES, ch * N_SAMPLES + chunkLen).set(mix[ch].subarray(start, end));

			var inputTensor = new ort.Tensor('float32', chunkBuf, [1, 2, N_SAMPLES]);
			return session.run({ mix: inputTensor }).then(function (result) {
				var stems = result.stems.data;
				for (var rk = 0; rk < rows.length; ++rk) {
					var rowOffset = (rows[rk] * 2) * N_SAMPLES;
					var oo = outs[rows[rk]];
					for (var c2 = 0; c2 < 2; ++c2) {
						var rowStart = rowOffset + c2 * N_SAMPLES;
						for (var s2 = 0; s2 < chunkLen; ++s2)
							oo[c2][start + s2] += stems[rowStart + s2] * win[s2];
					}
				}
				for (var s3 = 0; s3 < chunkLen; ++s3) weight[start + s3] += win[s3];
				result.stems.dispose && result.stems.dispose();
				++i;
				onProgress && onProgress(i / nChunks, i, nChunks);
				// yield so the UI can paint between chunks
				return new Promise(function (r) { setTimeout(r, 0); }).then(step);
			});
		}
		return step();
	}

	function subtractStem ( mix, stem ) {
		var out = [new Float32Array(mix[0].length), new Float32Array(mix[0].length)];
		for (var c = 0; c < 2; ++c)
			for (var s = 0; s < mix[0].length; ++s)
				out[c][s] = mix[c][s] - stem[c][s];
		return out;
	}

	// ---------- main flow ----------

	function runSeparation ( stemIds, ui ) {
		var eng = engine();
		var ws = eng.wavesurfer;
		var buffer = ws.backend.buffer;

		var region = ws.regions.list[0];
		var start = region ? eng.TrimTo(region.start, 3) : 0;
		var dur = region ? eng.TrimTo(region.end - region.start, 3) : buffer.duration;

		app.fireEvent('RequestPause');

		getSession(function (phase, frac, received) {
			if (phase === 'download' && frac !== undefined)
				ui.progress(frac, 'Downloading AI model: ' + Math.round(frac * 100) + '% (' + (received / 1048576).toFixed(0) + ' MB)');
			else if (phase === 'download')
				ui.progress(0, 'Downloading AI model (166 MB, first run only)...');
			else if (phase === 'init')
				ui.progress(1, 'Initializing model...');
		}).then(function () {
			if (cancelled) throw new Error('Cancelled');
			ui.progress(0, 'Preparing audio...');
			return resampleTo44kStereo(extractSegment(buffer, start, dur));
		}).then(function (mix) {
			if (cancelled) throw new Error('Cancelled');

			// which model rows do we actually need?
			var rowMap = {};
			for (var i = 0; i < stemIds.length; ++i)
				rowMap[stemIds[i] === 'accompaniment' ? 'vocals' : stemIds[i]] = 1;
			var rows = [];
			for (var k in rowMap) rows.push(STEM_ROW[k]);

			return separateRows(mix, rows, function (frac, i2, n) {
				ui.progress(frac, 'Running AI model: chunk ' + i2 + '/' + n + '...');
			}).then(function (rowOuts) {
				var stems = {};
				for (var j = 0; j < stemIds.length; ++j) {
					var id = stemIds[j];
					stems[id] = id === 'accompaniment' ?
						subtractStem(mix, rowOuts[STEM_ROW.vocals]) :
						rowOuts[STEM_ROW[id]];
				}
				return stems;
			});
		}).then(function (stems) {
			if (cancelled) throw new Error('Cancelled');
			ui.progress(1, 'Finalizing...');

			var buffers = {};
			var chain = Promise.resolve();
			stemIds.forEach(function (id) {
				chain = chain.then(function () {
					return stemToBuffer(stems[id], buffer.sampleRate, buffer.numberOfChannels)
						.then(function (b) { buffers[id] = b; });
				});
			});
			return chain.then(function () { return buffers; });
		}).then(function (buffers) {
			var h = host();

			if (stemIds.length === 1) {
				// single stem: load into the editor, with undo support
				var id = stemIds[0];
				var label = id.charAt(0).toUpperCase() + id.slice(1);

				app.fireEvent('StateRequestPush', {
					desc : 'AI Stem :: ' + label,
					meta : [ start, dur ],
					data : buffer
				});

				if (region) h.Replace(start, dur, buffers[id]);
				else h.FullReplace(buffers[id]);

				ws.regions.clear();
				ui.done('Done :: extracted ' + label);
				OneUp('AI Stem :: ' + label + ' extracted');
				return ;
			}

			// multi stem: export each stem as a WAV download
			var i = 0;
			var dl = function () {
				if (i >= stemIds.length) {
					ui.done('Done :: exported ' + stemIds.length + ' stems');
					OneUp('AI Stem :: exported ' + stemIds.length + ' stems');
					return ;
				}
				var id = stemIds[i];
				ui.progress(i / stemIds.length, 'Exporting stem-' + id + '.wav...');
				h.DownloadFile('stem-' + id + '.wav', 'wav', 0, false, true, 16, false, function () {}, buffers[id]);
				++i;
				setTimeout(dl, 500);
			};
			dl();
		}).catch(function (err) {
			if (err && err.message === 'Cancelled') return ;
			console.error(err);
			ui.error('Error: ' + (err && err.message ? err.message : err));
		});
	}

	// ---------- modal UI ----------

	app.listenFor('RequestActionFXUI_StemAI', function () {
		var eng = engine();
		if (!eng) return ;

		var mt = app.multitrack;
		if (mt && mt.IsOn && mt.IsOn())
			return OneUp('AI Stem Separation is not available in multitrack', 1500);
		if (!eng.is_ready || !eng.wavesurfer.backend.buffer)
			return OneUp('Load audio first', 1200);

		var modal_name = 'modalstemai';
		var running = false;

		var getStems = function ( q ) {
			var out = [];
			var boxes = q.el_body.getElementsByClassName('pk_check');
			for (var i = 0; i < boxes.length; ++i)
				if (boxes[i].checked) out.push(boxes[i].value);
			return out;
		};

		var body = '<div class="pk_row"><label>Stems to extract (tick one or more)</label>';
		for (var i = 0; i < STEM_LIST.length; ++i) {
			body += '<input type="checkbox" class="pk_check" id="pkai' + i + '" value="' +
				STEM_LIST[i].id + '"' + (i === 0 ? ' checked' : '') + '>' +
				'<label for="pkai' + i + '">' + STEM_LIST[i].label + '</label>';
		}
		body += '</div>' +
			'<div class="pk_row pk_inact" style="border:none">Runs fully in your browser (HT-Demucs AI model).<br>' +
			'First run downloads ~166 MB model (cached afterwards).<br>' +
			'One inference pass produces all stems — ticking more stems costs no extra time.<br>' +
			'A single stem loads into the editor (undo-able); multiple stems are exported as WAV downloads.</div>' +
			'<div class="pk_row" style="border:none">' +
			'<div class="pk_stemai_track" style="height:6px;background:#222;border-radius:3px;overflow:hidden">' +
			'<div class="pk_stemai_fill" style="height:100%;width:0%;background:#4a9d5b;transition:width .2s"></div></div>' +
			'<div class="pk_stemai_status pk_inact" style="margin-top:6px;min-height:16px"></div></div>';

		var x = new PKSimpleModal({
			title : 'AI Stem Separation (Beta)',

			ondestroy : function ( q ) {
				if (running) cancelled = true;
				app.ui.InteractionHandler.on = false;
				app.ui.KeyHandler.removeCallback(modal_name + 'esc');
			},

			buttons : [
				{
					title : 'Separate',
					clss : 'pk_modal_a_accpt',
					callback : function ( q ) {
						if (running) return ;
						var ids = getStems(q);
						if (!ids.length) return OneUp('Pick at least one stem', 1200);

						running = true;
						cancelled = false;
						q.els.bottom[0].classList.add('pk_inact');

						runSeparation(ids, {
							progress : function (frac, txt) {
								if (!q.el) return ;
								q.el_body.getElementsByClassName('pk_stemai_fill')[0].style.width = Math.round(frac * 100) + '%';
								q.el_body.getElementsByClassName('pk_stemai_status')[0].textContent = txt;
							},
							done : function (txt) {
								running = false;
								q.Destroy();
							},
							error : function (txt) {
								running = false;
								if (!q.el) return ;
								q.els.bottom[0].classList.remove('pk_inact');
								q.el_body.getElementsByClassName('pk_stemai_status')[0].textContent = txt;
							}
						});
					}
				}
			],

			body : body,

			setup : function ( q ) {
				app.fireEvent('RequestPause');
				app.ui.InteractionHandler.checkAndSet('modal');
				app.ui.KeyHandler.addCallback(modal_name + 'esc', function ( e ) {
					if (!app.ui.InteractionHandler.check('modal')) return ;
					q.Destroy();
				}, [27]);
			}
		});
		x.Show();
	});

	// ---------- menu entry (Effects menu) ----------

	var attempts = 0;
	function injectMenu () {
		if (!app.ui || !app.ui.el) {
			if (++attempts < 60) setTimeout(injectMenu, 250);
			return ;
		}

		var hdr = app.ui.el.getElementsByClassName('pk_hdr')[0];
		if (!hdr) {
			if (++attempts < 60) setTimeout(injectMenu, 250);
			return ;
		}

		var containers = hdr.children;
		for (var i = 0; i < containers.length; ++i) {
			var top_btn = containers[i].getElementsByTagName('button')[0];
			if (!top_btn || (top_btn.textContent || '').trim() !== 'Effects') continue;

			var menu = containers[i].getElementsByClassName('pk_menu')[0];
			if (!menu) return ;
			if (menu.getElementsByClassName('pk_stemai_item')[0]) return ;

			var sep = d.createElement('div');
			sep.className = 'pk_menu_el pk_stemai_item';
			var sepBtn = d.createElement('button');
			sepBtn.className = 'pk_opt';
			sepBtn.setAttribute('tab-index', '-1');
			sepBtn.textContent = '---';
			sep.appendChild(sepBtn);

			var cont = d.createElement('div');
			cont.className = 'pk_menu_el pk_stemai_item';
			var btn = d.createElement('button');
			btn.className = 'pk_opt';
			btn.setAttribute('tab-index', '-1');
			btn.textContent = 'AI Stem Separation (Beta)';
			btn.onclick = function () {
				if (app.ui.TopHeader && app.ui.TopHeader.closeMenu)
					app.ui.TopHeader.closeMenu();
				app.fireEvent('RequestActionFXUI_StemAI');
			};
			cont.appendChild(btn);

			menu.appendChild(sep);
			menu.appendChild(cont);
			return ;
		}

		if (++attempts < 60) setTimeout(injectMenu, 250);
	}

	setTimeout(injectMenu, 350);

})( window, document, PKAudioEditor );
