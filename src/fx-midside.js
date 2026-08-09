/*
 * AudioMass — Mid/Side Extractor (Vocal / Accompaniment)
 * Self-contained feature module: registers its own FX, events, modal and menu entry.
 * Works on the stereo image: Vocal = Mid (center), Accompaniment = Side.
 */
(function ( w, d, PKAE ) {
	'use strict';

	var app = PKAE;

	function engine () { return app.engine; }
	function host () { return app.engine && app.engine.FXPreviewHost; }

	// ---- FX registration (buffer-domain, same pattern as Invert/Flip) ----
	function registerFX () {
		var h = host ();
		if (!h || !h.FXBank || h.FXBank.MidSide) return ;

		h.FXBank.MidSide = function ( val ) {
			var mode = val && val.mode === 'side' ? 'side' : 'vocal';
			var res = val && val.residual !== undefined ?
				Math.max (0, Math.min (1, val.residual / 1)) : 0;

			return {
				filter : function ( audio_ctx, destination, source, duration ) {
					var buffer = source.buffer;
					if (buffer && buffer.numberOfChannels === 2) {
						var L = buffer.getChannelData (0);
						var R = buffer.getChannelData (1);
						var len = buffer.length;
						var i, mid, side;

						if (mode === 'vocal') {
							// keep center, optionally blend back some of the sides
							for (i = 0; i < len; ++i) {
								mid  = (L[i] + R[i]) * 0.5;
								side = (L[i] - R[i]) * 0.5 * res;
								L[i] = mid + side;
								R[i] = mid - side;
							}
						}
						else {
							// keep sides (karaoke), optionally blend back some center
							for (i = 0; i < len; ++i) {
								mid  = (L[i] + R[i]) * 0.5 * res;
								side = (L[i] - R[i]) * 0.5;
								L[i] =  side + mid;
								R[i] = -side + mid;
							}
						}
					}
					source.connect (destination);
					return (source);
				},
				update : function () {}
			};
		};
	}

	function modeLabel ( val ) {
		return val && val.mode === 'side' ? 'Accompaniment' : 'Vocal';
	}

	function isStereoReady () {
		var eng = engine ();
		return !!(eng && eng.is_ready && eng.wavesurfer.backend.buffer &&
			eng.wavesurfer.backend.buffer.numberOfChannels === 2);
	}

	function targetRegion () {
		var eng = engine ();
		var ws = eng.wavesurfer;
		var region = ws.regions.list[0];
		if (!region) {
			ws.regions.add ({ start:0, end:ws.getDuration (), id:'t' });
			region = ws.regions.list[0];
		}
		return [ eng.TrimTo (region.start, 3), eng.TrimTo (region.end - region.start, 3) ];
	}

	// ---- engine-level events (same conventions as engine.js) ----
	app.listenFor ('RequestActionFX_PREVIEW_MidSide', function ( val ) {
		registerFX ();
		var eng = engine ();
		if (!eng || !eng.is_ready) return ;

		var h = host ();
		if (h.previewing) {
			h.FXPreviewStop ();
			app.fireEvent ('DidStopPreview');
		}
		if (!isStereoReady ()) return OneUp ('Mid/Side needs a stereo file', 1600);

		var r = targetRegion ();
		h.FXPreview (r[0], r[1], h.FXBank.MidSide ( val ));
		app.fireEvent ('DidStartPreview');
	});

	app.listenFor ('RequestActionFX_MidSide', function ( val ) {
		registerFX ();
		var eng = engine ();
		if (!eng || !eng.is_ready) return ;
		if (!isStereoReady ()) return OneUp ('Mid/Side needs a stereo file', 1600);

		app.fireEvent ('RequestPause');

		var ws = eng.wavesurfer;
		var r = targetRegion ();

		app.fireEvent ('StateRequestPush', {
			desc : 'Mid/Side ' + modeLabel ( val ),
			meta : [ r[0], r[1] ],
			data : ws.backend.buffer
		});

		var h = host ();
		h.FX (r[0], r[1], h.FXBank.MidSide ( val ));

		OneUp ('Applied Mid/Side :: ' + modeLabel ( val ));
	});

	// ---- modal UI ----
	app.listenFor ('RequestActionFXUI_MidSide', function () {
		registerFX ();
		var eng = engine ();
		if (!eng) return ;

		var mt = app.multitrack;
		if (mt && mt.IsOn && mt.IsOn ())
			return OneUp ('Mid/Side is not available in multitrack', 1500);
		if (!eng.is_ready || !eng.wavesurfer.backend.buffer)
			return OneUp ('Load audio first', 1200);
		if (eng.wavesurfer.backend.buffer.numberOfChannels !== 2)
			return OneUp ('Mid/Side needs a stereo file', 1600);

		app.fireEvent ('RequestSelect', 1);

		var modal_name = 'modalms';
		var debounce = null;

		var getVal = function ( q ) {
			var mode = 'vocal';
			var radios = q.el_body.getElementsByClassName ('pk_check');
			for (var i = 0; i < radios.length; ++i)
				if (radios[i].checked) { mode = radios[i].value; break; }
			var range = q.el_body.getElementsByClassName ('pk_horiz')[0];
			return { mode: mode, residual: (range.value / 100) };
		};

		var x = new PKSimpleModal ({
			title : 'Extract Vocal / Accompaniment (Mid-Side)',

			ondestroy : function ( q ) {
				if (debounce) clearTimeout (debounce);
				app.fireEvent ('RequestActionFX_PREVIEW_STOP');
				app.ui.InteractionHandler.on = false;
				app.ui.KeyHandler.removeCallback (modal_name + 'esc');
			},

			toolbar : [
				{
					title : 'Preview',
					callback : function ( q ) {
						app.fireEvent ('RequestActionFX_PREVIEW_MidSide', getVal ( q ));
					}
				},
				{
					title : 'Stop Preview',
					callback : function ( q ) {
						app.fireEvent ('RequestActionFX_PREVIEW_STOP');
					}
				}
			],

			buttons : [
				{
					title : 'Apply',
					clss : 'pk_modal_a_accpt',
					callback : function ( q ) {
						app.fireEvent ('RequestActionFX_PREVIEW_STOP');
						app.fireEvent ('RequestActionFX_MidSide', getVal ( q ));
						q.Destroy ();
					}
				}
			],

			body :
				'<div class="pk_row"><label>Extract</label>' +
				'<input type="radio" class="pk_check" id="pkms1" name="pkmsmode" value="vocal" checked>' +
				'<label for="pkms1">Vocal (Mid / center)</label>' +
				'<input type="radio" class="pk_check" id="pkms2" name="pkmsmode" value="side">' +
				'<label for="pkms2">Accompaniment (Side / karaoke)</label></div>' +
				'<div class="pk_row" style="border:none"><label class="pk_line">Residual bleed</label>' +
				'<input type="range" class="pk_horiz" min="0" max="100" step="1" value="0" />' +
				'<span class="pk_val">0%</span></div>' +
				'<div class="pk_row pk_inact" style="border:none">Requires a stereo file with both L/R channels ON.<br>' +
				'Vocal = center of the stereo image, Accompaniment = the sides.</div>',

			setup : function ( q ) {
				var range = q.el_body.getElementsByClassName ('pk_horiz')[0];
				var span = q.el_body.getElementsByClassName ('pk_val')[0];
				var restart = function () {
					if (debounce) clearTimeout (debounce);
					debounce = setTimeout (function () {
						var h = host ();
						if (h && h.previewing)
							app.fireEvent ('RequestActionFX_PREVIEW_MidSide', getVal ( q ));
					}, 120);
				};

				range.oninput = function () {
					span.innerHTML = range.value + '%';
					restart ();
				};

				var radios = q.el_body.getElementsByClassName ('pk_check');
				for (var i = 0; i < radios.length; ++i)
					radios[i].onchange = restart;

				app.fireEvent ('RequestPause');
				app.ui.InteractionHandler.checkAndSet ('modal');
				app.ui.KeyHandler.addCallback (modal_name + 'esc', function ( e ) {
					if (!app.ui.InteractionHandler.check ('modal')) return ;
					q.Destroy ();
				}, [27]);
			}
		});
		x.Show ();
	});

	// ---- menu entry (Effects menu) ----
	var attempts = 0;
	function injectMenu () {
		if (!app.ui || !app.ui.el) {
			if (++attempts < 60) setTimeout (injectMenu, 250);
			return ;
		}

		var hdr = app.ui.el.getElementsByClassName ('pk_hdr')[0];
		if (!hdr) {
			if (++attempts < 60) setTimeout (injectMenu, 250);
			return ;
		}

		var containers = hdr.children;
		for (var i = 0; i < containers.length; ++i) {
			var top_btn = containers[i].getElementsByTagName ('button')[0];
			if (!top_btn || (top_btn.textContent || '').trim () !== 'Effects') continue;

			var menu = containers[i].getElementsByClassName ('pk_menu')[0];
			if (!menu) return ;
			if (menu.getElementsByClassName ('pk_ms_item')[0]) return ; // already injected

			var cont = d.createElement ('div');
			cont.className = 'pk_menu_el pk_ms_item';
			var btn = d.createElement ('button');
			btn.className = 'pk_opt';
			btn.setAttribute ('tab-index', '-1');
			btn.textContent = 'Vocal / Accompaniment (Mid-Side)';
			btn.onclick = function () {
				if (app.ui.TopHeader && app.ui.TopHeader.closeMenu)
					app.ui.TopHeader.closeMenu ();
				app.fireEvent ('RequestActionFXUI_MidSide');
			};
			cont.appendChild (btn);
			menu.appendChild (cont);
			return ;
		}

		if (++attempts < 60) setTimeout (injectMenu, 250);
	}

	setTimeout (injectMenu, 350);

})( window, document, PKAudioEditor );
