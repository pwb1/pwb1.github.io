

var map = d3.geomap.choropleth()
    .geofile('./scripts/d3-geomap/topojson/countries/USA.json')
    .projection(d3.geo.albersUsa)
    .column('Fighters')
    .unitId('fips')
    .scale(900)
    .legend(true)
    .zoomFactor(2)
    .colors(colorbrewer.YlGnBu[9]);

d3.csv('./data/map_data.csv', function(error, data) {
    d3.select('#map')
        .datum(data)
        .call(map.draw, map);
});