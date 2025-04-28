//VERSION=3
/*
MODIFIED: Only output [0,0,255] for water and [255,255,255] for non-water
SOURCE: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/simple_water_bodies_mapping-swbm/
*/

var source = "S2L2A";
var MNDWI_thr = 0.1;
var NDWI_thr = 0.2;
var SWI_thr = 0.03;

function wbiS2(g, nr, s1, vre1) {
    try {
        var mndwi = index(g, s1);
        var ndwi = index(g, nr);
        var swi = index(vre1, s1);
        if (mndwi > MNDWI_thr || ndwi > NDWI_thr || swi > SWI_thr) {
            return 1;
        }
    } catch (err) {}
    return 0;
}

function setup() {
    return {
        input: ["B02", "B03", "B04", "B05", "B08", "B11"],
        output: { bands: 3 }
    };
}

function evaluatePixel(p) {
    var g = p.B03;
    var nr = p.B08;
    var s1 = p.B11;
    var vre1 = p.B05;

    var water = wbiS2(g, nr, s1, vre1);

    if (water == 1) {
        return [0, 0, 255]; // Water: Blue
    } else {
        return [255, 255, 255]; // Non-water: White
    }
}
