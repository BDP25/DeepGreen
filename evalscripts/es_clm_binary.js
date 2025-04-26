// VERSION=3
// returns CLM as binary mask (black pixel = cloudy pixel)
function setup() {
    return {
        input: [{
            bands: ["B02", "B03", "B04", "CLM"]
        }],
        output: { 
            bands: 3 
        }
    };
  }
  
function evaluatePixel(sample) {
    if (sample.CLM == 1) {
        return [0, 0, 0]
        }
    return [255, 255, 255];
  }