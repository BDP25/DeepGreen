// VERSION=3
// all bands + CLM
// returns all bands of Sentinel-2 data along with the cloud mask (CLM).
function setup() {
    return {
        input: [{
            bands: [
                "B01", // Coastal aerosol     | 442.7 nm  | 60 m
                "B02", // Blue                | 492.4 nm  | 10 m
                "B03", // Green               | 559.8 nm  | 10 m
                "B04", // Red                 | 664.6 nm  | 10 m
                "B05", // Vegetation red edge | 704.1 nm  | 20 m
                "B06", // Vegetation red edge | 740.5 nm  | 20 m
                "B07", // Vegetation red edge | 782.8 nm  | 20 m
                "B08", // NIR                 | 832.8 nm  | 10 m
                "B8A", // Narrow NIR          | 864.7 nm  | 20 m
                "B09", // Water vapour        | 945.1 nm  | 60 m
                "B11", // SWIR                | 1613.7 nm | 20 m
                "B12", // SWIR                | 2202.4 nm | 20 m
                "CLM"] // Cloud Mask          | -         | 160 m 
        }],

        output: { 
            bands: 13,
        }
    };
  }
  
function evaluatePixel(sample) {
    return [
        sample.B01, 
        sample.B02, 
        sample.B03, 
        sample.B04, 
        sample.B05, 
        sample.B06, 
        sample.B07, 
        sample.B08, 
        sample.B8A,
        sample.B09,
        sample.B11,
        sample.B12,
        sample.CLM
    ];
  }