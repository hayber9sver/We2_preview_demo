# Himax / Grove Vision AI V2 — SSCMA serial protocol (reverse-engineered from web toolkit)

Source toolkit: `/home/orangepi/Himax_AI_web_toolkit/` (built Vue app).
Protocol logic lives in `assets/index-legacy.8f5c53d6.js` (serial transport + framing + AT commands)
and `assets/index-legacy.51f14f00.js` (preview/draw of boxes onto canvas).

Legend: **[JS-CONFIRMED]** = found directly in the minified JS. **[ASSUMED]** = SSCMA convention, not
contradicted by JS but not byte-proven here.

---

## a. Baud rate  — **921600**  [JS-CONFIRMED]
```
this.serial?.open({baudRate:921600})        // index-legacy.8f5c53d6.js, connect()
```
(115200 also appears in the same file but ONLY in the esptool/esploader flashing path — `romBaudrate=115200`,
esploader `connect(e=115200)`. That is for ROM bootloader flashing, NOT the runtime SSCMA link. We never flash.)

## b. Start-stream command sequence  [JS-CONFIRMED]
The toolkit, to begin the continuous inference-with-image live view, does:
1. `connect()`: open port @921600, start read loop, **hardReset()** = toggle RTS low/high (software reset via
   DTR/RTS, NOT a flash), wait 2000 ms, clear the input buffer.
   ```
   await this.hardReset(); await Bi(2e3); this.serial?.clear();
   hardReset(){ await setRTS(!1); await Bi(100); await setRTS(!0); }   // 100ms low pulse
   ```
2. Register an event listener for the `"INVOKE"` event, then call `invoke(-1)`:
   ```
   s.value?.addEventListener("INVOKE", re);     // 51f14f00.js  (re = draw callback)
   const e = await s.value?.invoke(-1);         // 51f14f00.js
   ```
3. `invoke(n)` builds the command string:
   ```
   invoke(e){ return `AT+INVOKE=${e},0,0\r` }   // 8f5c53d6.js  (note: trailing \r only)
   ```
   So the literal bytes sent to start the forever stream are:

   **`AT+INVOKE=-1,0,0\r`**   (ASCII, terminated by a single carriage-return `\r` = 0x0D)

   SSCMA arg meaning [ASSUMED for arg semantics, command string is JS-CONFIRMED]:
   `AT+INVOKE=<n_times>,<diff>,<result_only>`
   - `n_times = -1`  → invoke forever (continuous stream)
   - `diff = 0`      → emit a result for every frame (not only when detections change)
   - `result_only = 0` → 0 means **include the JPEG image** in each event. (=1 would be result-only/no image.)
     Empirically the toolkit's INVOKE events carry `data.image`, consistent with result_only=0.

   NOTE: Our pyserial port is opened by us; we do NOT rely on the RTS pulse reset. We send an `AT+BREAK\r`
   first to stop any prior stream, drain, then send `AT+INVOKE=-1,0,0\r`. (RST/reset is optional and we avoid
   it to keep the link simple; toggling RTS via pyserial is supported but not required to start the stream.)

Other AT commands the toolkit knows (NOT needed for live view, listed for reference; many are query-only):
`AT+RST\r` (soft reset), `AT+ID?`, `AT+NAME?`, `AT+VER?`, `AT+STAT?`, `AT+MODEL?`, `AT+MODELS?`,
`AT+ALGOS?`, `AT+SENSOR?/=`, `AT+SAMPLE=<n>` (sample image only, no inference — NOT used for the AI live view),
`AT+TSCORE?`, `AT+TIOU?`, `AT+ACTION=`, `AT+INFO=`, `AT+BREAK` (stop), `AT+LED=`.
None of these write flash. We only ever use INVOKE and BREAK.

## c. Incoming packet framing  [JS-CONFIRMED]
Read loop (8f5c53d6.js) scans the raw byte stream with a small state machine. Byte codes:
`123='{'  125='}'  13='\r'  10='\n'`.
```
for each byte i:
  if i==123 ('{'):
     if lastCode==13 ('\r'):  hasStart=true; push '{'        // packet starts at  \r{
     elif hasStart:           push '{'
  elif i==10 ('\n'):
     if lastCode==125 ('}'):  hasStart=false; emit cacheData  // packet ends at   }\n
     elif hasStart:           push '\n'
  else:
     if hasStart: push i
  lastCode = i
```
So a complete packet is the JSON text starting at a `{` that was immediately preceded by `\r`, up to and
including the `}` that is immediately followed by `\n`. i.e. each response is wrapped as `\r{ ... }\n`.
- Packets CAN be split across reads — the state machine accumulates across reads (`cacheData` persists).  [JS-CONFIRMED]
- The collected bytes are JSON-decoded: `JSON.parse(textDecoder.decode(cacheData))`.  [JS-CONFIRMED]
- Boot text / AT echoes / log lines that aren't `\r{...}\n` are naturally ignored by the state machine.

Our Python port replicates this exact state machine byte-for-byte (start = prev=='\r' & cur=='{',
end = prev=='}' & cur=='\n').

## d. Response / data schema  [JS-CONFIRMED for fields used]
Parsed object top level:
```
{ "type": <int>, "name": "<EVENT>", "code": <int>, "data": { ... } }
```
- `type == 0` → a direct reply to a command we sent (looked up in resolveMap by name).
- `type == 1` → an asynchronous **event** (looked up in eventMap by name). The live stream frames are
  `type==1, name=="INVOKE"`.
  ```
  s=JSON.parse(...); n=s.type; o=s.name;
  if(0===n){ resolveMap.get(o)(s) } else if(1===n){ eventMap.get(o)(s) ... }
  ```
`data` object for an INVOKE event contains (51f14f00.js draw callback reads):
- `data.image`  → **base64-encoded JPEG**, used as `W.src = "data:image/jpg;base64,"+image`.  [JS-CONFIRMED]
- `data.boxes`  → array of detection boxes. Each box is a 6-element array:
  **`[x, y, w, h, score, target]`**  [JS-CONFIRMED]
  ```
  const t=r[0]  // x   (top-left, see scaling below)
        o=r[1]  // y   (top-left)
        i=r[2]  // w
        s=r[3]  // h
        l=r[4]  // score
        n=parseInt(r[5],10)  // target / class id
  ...strokeRect(t,o,i,s)      // canvas strokeRect(x,y,w,h)  => x,y is TOP-LEFT corner
  ...fillText(`${classname}: ${score}`, t+5, o+15)
  ```
  **Coordinate convention: x,y = TOP-LEFT corner; w,h = width,height** (because HTML canvas
  `strokeRect(x,y,w,h)` interprets the first two args as the top-left). [JS-CONFIRMED via strokeRect usage]
- `data.classes` / model classes: the human label is `currentModel.classes[target]` (array of names);
  color = palette[target % palette.length]. If no class name, the raw id is shown. [JS-CONFIRMED]
- `data.resolution` → present per SSCMA (`[w,h]`); the toolkit mainly relies on the decoded JPEG's own
  width/height (`W.width/W.height`) rather than `resolution` for scaling. [resolution field ASSUMED present]
- `data.perf` / the toolkit reads `data.algo_tick` ( `FPS = 4e8 / algo_tick[0]` ). `perf` is the SSCMA name
  (`[prepare_ms, inference_ms, postprocess_ms]`). [algo_tick JS-CONFIRMED; perf ASSUMED]
- Other optional box arrays exist for special models: `peoplenet_boxes`, `fm_face_boxes`, `gender_cls_boxes`,
  plus `points` for pose/keypoint models. For generic detection models, `boxes` is the one to use. [JS-CONFIRMED names]

## e. Image decode + box scaling on display  [JS-CONFIRMED]
```
W.onload = () => {
  if (W.width<640 || W.height<480) {            // small frame (e.g. 240x240, 416x416)
     i = 3;                                      // SCALE FACTOR = 3
     canvas.width  = W.width *(i+0.5);           // canvas sized 3.5x  (extra room)
     canvas.height = W.height*(i+0.5);
     ctx.drawImage(W, 0,0, W.width*i, W.height*i);   // image drawn at 3x
  } else {                                       // already large frame
     canvas.width = W.height; canvas.height = W.width;
     ctx.drawImage(W, 0,0, W.width, W.height);        // i stays 1
  }
  // then for each box: r[0..3] *= i   (scale x,y,w,h by the SAME factor the image was drawn at)
}
W.src = "data:image/jpg;base64," + data.image;
```
So box coords come back in the JPEG's native pixel space, and the toolkit multiplies x,y,w,h by the display
scale `i` (3 for small images, 1 for large) so the boxes line up with the up-scaled image. In our Python app
we draw on the decoded frame at its native resolution, so **boxes map 1:1 with no scaling needed** (we keep the
JPEG at native size). If we later upscale the frame for viewing we apply the same factor to the boxes.

---
## Summary of facts the app hardcodes
- BAUD = 921600
- START   = `AT+INVOKE=-1,0,0\r`  (bytes: 41 54 2B 49 4E 56 4F 4B 45 3D 2D 31 2C 30 2C 30 0D)
- STOP    = `AT+BREAK\r`
- Packet  = JSON wrapped as `\r{ ... }\n`; accumulate across reads; state machine start `\r{`, end `}\n`.
- Event of interest: `type==1, name=="INVOKE"`, fields `data.image` (b64 jpeg), `data.boxes` ([x,y,w,h,score,target], x/y top-left).
