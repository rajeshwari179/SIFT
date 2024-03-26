const rgbToHslHsvHex = (rgb) => {
  var rgbArr = [rgb.r, rgb.g, rgb.b];
  var M, m, C, hue, V, L, Sv, Sl;
  M = Math.max(...rgbArr);
  m = Math.min(...rgbArr);
  C = M - m;
  // I = (rgbArr[0] + rgbArr[1] + rgbArr[2]) / 3;
  // Hue
  if (C === 0) hue = 0;
  else if (M === rgbArr[0]) hue = ((rgbArr[1] - rgbArr[2]) / C) % 6;
  else if (M === rgbArr[1]) hue = (rgbArr[2] - rgbArr[0]) / C + 2;
  else if (M === rgbArr[2]) hue = (rgbArr[0] - rgbArr[1]) / C + 4;
  hue *= 60;
  // Lightness and Value
  V = M / 255;
  L = (M + m) / (2 * 255);
  // Saturation
  if (V === 0) Sv = 0;
  else Sv = C / (V * 255);
  if (L === 1 || L === 0) Sl = 0;
  else Sl = C / (255 * (1 - Math.abs(2 * L - 1)));

  hue = ((hue % 360) + 360) % 360;
  let hsv = { h: hue, s: Sv, v: V, a: 1 };
  let hsl = { h: hue, s: Sl, l: L, a: 1 };
  rgb.a = 1;
  let hex = '#';
  for (let i in rgbArr) {
    let colorcode = Math.floor(rgbArr[i]).toString(16);
    hex += '0'.repeat(2 - colorcode.length) + colorcode;
  }
  return { rgb: rgb, hsv: hsv, hsl: hsl, hex: hex };
};

/**
 * Function that converts hex encoded colour to rgb format.
 *
 * @param {string} hex
 * @returns {rgb} {r,g,b}
 */
const hexToRgb = (hex) => {
  var aRgbHex = hex.match(/.{1,2}/g);
  // console.log(aRgbHex)
  var aRgb = {
    r: parseInt(aRgbHex[0], 16),
    g: parseInt(aRgbHex[1], 16),
    b: parseInt(aRgbHex[2], 16),
  };
  return aRgb;
};

/**
 * Function to convert multi-representation colour object to 2x3 array of RGB and HSV representations.
 *
 * @param {{rgb:object, hsv:object, [hsl:object, hex:object]}} obj
 * @return {[Array, Array]} [[h, s, v], [r, g, b]].
 */
const hsvRgbObjToArr = (obj) => {
  var arr = [
    [0, 0, 0],
    [0, 0, 0],
  ];
  arr[0] = [obj.hsv.h, obj.hsv.s, obj.hsv.v];
  arr[1] = [obj.rgb.r, obj.rgb.g, obj.rgb.b];
  return arr;
};

const getColourPoints = (rawCP) => {
  let cP = new Array(rawCP.radii.length);
  for (let i in rawCP.colours) {
    // console.log(rawCP.colours[i]);
    // console.log(hexToRgb(rawCP.colours[i]));
    cP[i] = {
      x: rawCP.positions[i][0] / rawCP.viewport[0],
      y: rawCP.positions[i][1] / rawCP.viewport[1],
      colour: rgbToHslHsvHex(hexToRgb(rawCP.colours[i])),
      colourArr: hsvRgbObjToArr(rgbToHslHsvHex(hexToRgb(rawCP.colours[i]))),
      radius: rawCP.radii[i],
    };
  }
  // console.log(cP);
  return cP;
};

export default rgbToHslHsvHex;
export { hexToRgb, hsvRgbObjToArr, getColourPoints };
