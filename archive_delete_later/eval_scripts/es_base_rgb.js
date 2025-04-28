// VERSION=3
// returns RGB image with all bands multiplied by 4.5 for brightness
function setup() {
    return {
      input: ["B02", "B03", "B04"],
      output: { bands: 3 }
    };
  }
  
  function evaluatePixel(sample) {
    return [4.5 * sample.B04, 4.5 * sample.B03, 4.5 * sample.B02];
  }