anychart.onDocumentReady(function () {
  // code = '005930';
  code = localStorage.getItem("stock_code");
  localStorage.removeItem('stock_code');
  anychart.data.loadCsvFile(
    // 'https://gist.githubusercontent.com/shacheeswadia/cd509e0b0c03964ca86ae7d894137043/raw/5f336c644ad61728dbac93026f3268b86b8d0680/teslaDailyData.csv',
    'static/'+ code + '.csv',
    function (data) {
    // create data table on loaded data
    const num_data = data.split('\n').length;

    localStorage.setItem("tomorrow_high", data.split('\n')[num_data-6].split(",")[2]);
    localStorage.setItem("tomorrow_low", data.split('\n')[num_data-6].split(",")[3]);
    localStorage.setItem("today_high", data.split('\n')[num_data-7].split(",")[2]);
    localStorage.setItem("today_low", data.split('\n')[num_data-7].split(",")[3]);
    
    line_data = data.split('\n').slice(0, num_data-6).join('\n');
    

    var dataTable = anychart.data.table();
    dataTable.addData(data);

    var lineTable = anychart.data.table();
    lineTable.addData(line_data);

    // map loaded data for the candlestick series
    var mapping = lineTable.mapAs({
      open: 1,
      high: 2,
      low: 3,
      close: 4
    });


    // create stock chart
    var chart = anychart.stock();
  
    // create first plot on the chart
    var plot = chart.plot(0);



    // Add a range spline series using the data from the computer
    var bbandsSeries = chart.plot().rangeSplineArea(dataTable, {
      high: 2,
      low: 3
    });

    // Set name and visual effects
    bbandsSeries
    .name('Bands')
    .fill(anychart.color.lighten('#CDB4DB', 0.7))
    .highStroke('#FF7F7F')
    .lowStroke('#ADD8E6');

    // set grid settings
    plot.yGrid(true).xGrid(true).yMinorGrid(true).xMinorGrid(true);

    var series = plot.candlestick(mapping).name('SAMSUMG ELECTRONICS CO.');
    series.legendItem().iconType('rising-falling');
    series.risingFill("#E34234");
    series.risingStroke("#E34234");
    series.fallingFill("#1E90FF");
    series.fallingStroke("#1E90FF");
    
    // create scroller series with mapped data
    chart.scroller().candlestick(mapping);
    
    // set chart selected date/time range
    chart.selectRange('2023-11-01', '2023-11-30');
    
    // create range picker
    // var rangePicker = anychart.ui.rangePicker();
    
    // init range picker
    // rangePicker.render(chart);
    
    // create range selector
    var rangeSelector = anychart.ui.rangeSelector();
    
    // init range selector
    rangeSelector.render(chart);
    
    // sets the title of the chart
    // chart.title('Tesla Inc. Stock Chart');
  
    // set container id for the chart
    chart.container('container');
    
    // initiate chart drawing
    chart.draw();
    }
  );
});