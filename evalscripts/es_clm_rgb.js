// VERSION=3
// returns CLM as overlay on rgb (red)
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
        return [1 + sample.B04, sample.B03, sample.B02]
        }
    return [3.5*sample.B04, 3.5*sample.B03, 3.5*sample.B02];
  }