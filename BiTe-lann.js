// Construct Problem-specific Aritifical Neural Networks.
// This is for prediction of thermoelectric properties of BiTe-based materials.
// Programmed by Dr. Jaywan Chung
// v0.2a updated on Sep 17, 2023

"use strict";

const jcApp = {
    chartHeight: 500,
    chartWidth: 500,
    minTemp: 0,
    maxTemp: 300,
    nTempNodes: 100,
    dataLegend: 'Expt',
    plot1Legend: 'Pred 1',
    plot2Legend: 'Pred 2',
    colorRawData: '#594D5B',
    colorPlot1: '#808080',  // gray
    colorPlot2: '#1976D2',  // blue
};

class BiTeMeanLann {
    constructor(embeddingNet, meanDictionaryNet) {
        this.lann = new LatentSpaceNeuralNetwork(embeddingNet, meanDictionaryNet);
        this.outputMatrix = null;
    }
    evaluate(inputMatrix) {
        const scaledInput = BiTeLann.getScaledInput(inputMatrix);
        this.lann.evaluate(scaledInput);
        this.outputMatrix = this.lann.outputMatrix;
        this.scaleOutput();
    }
    static getScaledInput(inputMatrix) {
        const scaledInput = inputMatrix.clone();
        scaledInput.array[5] /= 300.0;  // scale temperature

        return scaledInput;
    }
    scaleOutput() {
        const y0 = this.outputMatrix.getElement(0, 0);
        const y2 = this.outputMatrix.getElement(2, 0);
        this.outputMatrix.array[0] = Math.log(Math.exp(y0) + 1) * 1e-05;  // softplus activation
        this.outputMatrix.array[1] *= 1e-04;
        this.outputMatrix.array[2] = Math.log(Math.exp(y2) + 1);
    }
}
class BiTeLann {
    constructor(embeddingNet, meanDictionaryNet, stdDictionaryNet) {
        this.meanLann = new LatentSpaceNeuralNetwork(embeddingNet, meanDictionaryNet);
        this.stdLann = new LatentSpaceNeuralNetwork(embeddingNet, stdDictionaryNet);
        this.meanOutputMatrix = null;
        this.stdOutputMatrix = null;
    }
    evaluate(inputMatrix) {
        const scaledInput = BiTeLann.getScaledInput(inputMatrix);
        this.meanLann.evaluate(scaledInput);
        this.stdLann.evaluate(scaledInput);
        this.meanOutputMatrix = this.meanLann.outputMatrix;
        this.stdOutputMatrix = this.stdLann.outputMatrix;
        this.scaleOutput();
    }
    static getScaledInput(inputMatrix) {
        const scaledInput = inputMatrix.clone();
        scaledInput.array[5] /= 300.0;  // scale temperature

        return scaledInput;
    }
    scaleOutput() {
        const y0 = this.meanOutputMatrix.getElement(0, 0);
        const y2 = this.meanOutputMatrix.getElement(2, 0);
        this.meanOutputMatrix.array[0] = Math.log(Math.exp(y0) + 1) * 1e-05;  // softplus activation
        this.meanOutputMatrix.array[1] *= 1e-04;
        this.meanOutputMatrix.array[2] = Math.log(Math.exp(y2) + 1);
        this.stdOutputMatrix.array[0] *= 1e-05;
        this.stdOutputMatrix.array[1] *= 1e-04;
    }
}

jcApp.startApp = function() {
    console.log("Starting App...");
    jcApp.initSelectRawdata();
    jcApp.initLann();

    jcApp.tempArray = jcApp.getLinearSpace(jcApp.minTemp, jcApp.maxTemp, jcApp.nTempNodes);
    jcApp.plot1Input = new Matrix(6, 1);  // x, p-type (0 or 1), n-type (0 or 1), a-axis (0 or 1), c-axis (0 or 1)
    jcApp.plot1ElecResiArray = new Float64Array(jcApp.nTempNodes);
    jcApp.plot1SeebeckArray = new Float64Array(jcApp.nTempNodes);
    jcApp.plot1ThrmCondArray = new Float64Array(jcApp.nTempNodes);
    jcApp.plot1ElecResiStdArray = new Float64Array(jcApp.nTempNodes);
    jcApp.plot1SeebeckStdArray = new Float64Array(jcApp.nTempNodes);
    jcApp.plot1ThrmCondStdArray = new Float64Array(jcApp.nTempNodes);
    jcApp.plot2Input = new Matrix(6, 1);
    jcApp.plot2ElecResiArray = new Float64Array(jcApp.nTempNodes);
    jcApp.plot2SeebeckArray = new Float64Array(jcApp.nTempNodes);
    jcApp.plot2ThrmCondArray = new Float64Array(jcApp.nTempNodes);
    jcApp.plot2ElecResiStdArray = new Float64Array(jcApp.nTempNodes);
    jcApp.plot2SeebeckStdArray = new Float64Array(jcApp.nTempNodes);
    jcApp.plot2ThrmCondStdArray = new Float64Array(jcApp.nTempNodes);
    console.log('Memory allocated.');

    google.charts.load('current', {'packages':['corechart']});
    google.charts.setOnLoadCallback(jcApp.activateChartsAndButtons); // activate buttons when google charts is loaded.

    // adjust style for mobile
    if (window.innerWidth <= 768) {
        jcApp.chartWidth = window.innerWidth * 0.9;
        jcApp.chartHeight = window.innerWidth * 0.9;
    }
}

jcApp.initSelectRawdata = function() {
    jcApp.select = document.getElementById("select-rawdata");
    for (const key of Object.keys(jcApp.rawdata)) {
        let opt = document.createElement("option");
        opt.value = key;
        opt.innerHTML = key;
        jcApp.select.appendChild(opt);
    }
    const copyToPlota1Button = document.getElementById("copy-to-plot1-button");
    copyToPlota1Button.addEventListener("click", jcApp.onClickCopyToPlot1Button);
    // select the first data
    // jcApp.select.options[0].selected = true;
    jcApp.select.value = 'n-type, x=0.50, a-axis';
    jcApp.onClickCopyToPlot1Button();
    console.log("'Select Data' initialized.");
}
jcApp.onClickCopyToPlot1Button = function() {
    let dataName = jcApp.select.value;
    if (!dataName) return;  // if not selected, do nothing.
    let input = jcApp.rawdata[dataName]["input"];
    document.getElementById("plot1-composition").value = input[0];
    if (input[1]) {
        document.getElementById("plot1-type").value = "p-type";
    } else {
        document.getElementById("plot1-type").value = "n-type";
    }
    if (input[3]) {
        document.getElementById("plot1-axis").value = "a-axis";
    } else {
        document.getElementById("plot1-axis").value = "c-axis";
    }
}
jcApp.activateChartsAndButtons = function() {
    jcApp.initTepCharts();

    document.getElementById("predict-tep").addEventListener("click", function() {
        jcApp.predict();
        jcApp.drawCharts();    
    });
}
jcApp.predict = function() {
    jcApp.clearPrediction();

    const plot1Composition = parseFloat(document.getElementById("plot1-composition").value);
    const plot1Type = document.getElementById("plot1-type").value;
    const plot1Axis = document.getElementById("plot1-axis").value;
    const plot2Composition = parseFloat(document.getElementById("plot2-composition").value);
    const plot2Type = document.getElementById("plot2-type").value;
    const plot2Axis = document.getElementById("plot2-axis").value;

    if (Number.isFinite(plot1Composition)) {
        jcApp.plot1Input.setElement(0, 0, plot1Composition);
        if (plot1Type == 'p-type') {
            jcApp.plot1Input.setElement(1, 0, 1);
            jcApp.plot1Input.setElement(2, 0, 0);
        } else {
            jcApp.plot1Input.setElement(1, 0, 0);
            jcApp.plot1Input.setElement(2, 0, 1);
        }
        if (plot1Axis == 'a-axis') {
            jcApp.plot1Input.setElement(3, 0, 1);
            jcApp.plot1Input.setElement(4, 0, 0);
        } else {
            jcApp.plot1Input.setElement(3, 0, 0);
            jcApp.plot1Input.setElement(4, 0, 1);
        }
        for(let i=0; i<jcApp.nTempNodes; i++) {
            jcApp.plot1Input.setElement(5, 0, jcApp.tempArray[i]);
            jcApp.lann.evaluate(jcApp.plot1Input);
            jcApp.plot1ElecResiArray[i] = jcApp.lann.meanOutputMatrix.array[0];
            jcApp.plot1SeebeckArray[i] = jcApp.lann.meanOutputMatrix.array[1];
            jcApp.plot1ThrmCondArray[i] = jcApp.lann.meanOutputMatrix.array[2];
            jcApp.plot1ElecResiStdArray[i] = jcApp.lann.stdOutputMatrix.array[0];
            jcApp.plot1SeebeckStdArray[i] = jcApp.lann.stdOutputMatrix.array[1];
            jcApp.plot1ThrmCondStdArray[i] = jcApp.lann.stdOutputMatrix.array[2];
        }
    }
    if (Number.isFinite(plot2Composition)) {
        jcApp.plot2Input.setElement(0, 0, plot2Composition);
        if (plot2Type == 'p-type') {
            jcApp.plot2Input.setElement(1, 0, 1);
            jcApp.plot2Input.setElement(2, 0, 0);
        } else {
            jcApp.plot2Input.setElement(1, 0, 0);
            jcApp.plot2Input.setElement(2, 0, 1);
        }
        if (plot2Axis == 'a-axis') {
            jcApp.plot2Input.setElement(3, 0, 1);
            jcApp.plot2Input.setElement(4, 0, 0);
        } else {
            jcApp.plot2Input.setElement(3, 0, 0);
            jcApp.plot2Input.setElement(4, 0, 1);
        }
        for(let i=0; i<jcApp.nTempNodes; i++) {
            jcApp.plot2Input.setElement(5, 0, jcApp.tempArray[i]);
            jcApp.lann.evaluate(jcApp.plot2Input);
            jcApp.plot2ElecResiArray[i] = jcApp.lann.meanOutputMatrix.array[0];
            jcApp.plot2SeebeckArray[i] = jcApp.lann.meanOutputMatrix.array[1];
            jcApp.plot2ThrmCondArray[i] = jcApp.lann.meanOutputMatrix.array[2];
            jcApp.plot2ElecResiStdArray[i] = jcApp.lann.stdOutputMatrix.array[0];
            jcApp.plot2SeebeckStdArray[i] = jcApp.lann.stdOutputMatrix.array[1];
            jcApp.plot2ThrmCondStdArray[i] = jcApp.lann.stdOutputMatrix.array[2];
        }
    }
    console.log("Prediction complete.");
}
jcApp.clearPrediction = function() {
    jcApp.plot1Input.fill(NaN);
    jcApp.plot1ElecResiArray.fill(NaN);
    jcApp.plot1SeebeckArray.fill(NaN);
    jcApp.plot1ThrmCondArray.fill(NaN);
    jcApp.plot1ElecResiStdArray.fill(NaN);
    jcApp.plot1SeebeckStdArray.fill(NaN);
    jcApp.plot1ThrmCondStdArray.fill(NaN);
    jcApp.plot2Input.fill(NaN);
    jcApp.plot2ElecResiArray.fill(NaN);
    jcApp.plot2SeebeckArray.fill(NaN);
    jcApp.plot2ThrmCondArray.fill(NaN);
    jcApp.plot2ElecResiStdArray.fill(NaN);
    jcApp.plot2SeebeckStdArray.fill(NaN);
    jcApp.plot2ThrmCondStdArray.fill(NaN);

    console.log("Prediction cleared.");
}
jcApp.checkShowOptions = function() {
    jcApp.showData = document.getElementById("show-data").checked;
    jcApp.showPlot1 = document.getElementById("show-plot1").checked;
    jcApp.showPlot1TepCi = document.getElementById("show-plot1-tep-ci").checked;
    jcApp.showPlot1zTCi = document.getElementById("show-plot1-zT-ci").checked;
    jcApp.showPlot2 = document.getElementById("show-plot2").checked;
    jcApp.showPlot2TepCi = document.getElementById("show-plot2-tep-ci").checked;
    jcApp.showPlot2zTCi = document.getElementById("show-plot2-zT-ci").checked;
}
jcApp.initTepCharts = function() {
    jcApp.chartElecResi = new google.visualization.ComboChart(document.getElementById('chart-elec-resi'));
    jcApp.chartSeebeck = new google.visualization.ComboChart(document.getElementById('chart-seebeck'));
    jcApp.chartThrmCond = new google.visualization.ComboChart(document.getElementById('chart-thrm-cond'));
    jcApp.chartElecCond = new google.visualization.ComboChart(document.getElementById('chart-elec-cond'));
    jcApp.chartPowerFactor = new google.visualization.ComboChart(document.getElementById('chart-power-factor'));
    jcApp.chartFigureOfMerit = new google.visualization.ComboChart(document.getElementById('chart-figure-of-merit'));
    console.log("Charts initialized.");
}

jcApp.drawCharts = function() {
    jcApp.checkShowOptions();
    jcApp.drawElecResiChart();
    jcApp.drawSeebeckChart();
    jcApp.drawThrmCondChart();
    jcApp.drawElecCondChart();
    jcApp.drawPowerFactorChart();
    jcApp.drawFigureOfMeritChart();
}
jcApp.drawTepChart = function(chart, yLabel, yScale, getTepData, getPlot1TepAndCi, getPlot2TepAndCi) {
    const xLabel = "Temperature (°C)";

    let data = new google.visualization.DataTable();
    data.addColumn('number', xLabel);
    data.addColumn('number', jcApp.dataLegend);
    data.addColumn('number', jcApp.plot1Legend);
    data.addColumn({type: 'number', role: 'interval'});
    data.addColumn({type: 'number', role: 'interval'});
    data.addColumn('number', jcApp.plot2Legend);
    data.addColumn({type: 'number', role: 'interval'});
    data.addColumn({type: 'number', role: 'interval'});

    let [tempData, tepData] = getTepData();
    if(jcApp.showData && (tempData !== null) && (tepData !== null)) {
        for(let i=0; i<tempData.length; i++) {
            data.addRow([tempData[i], tepData[i]*yScale, NaN, NaN, NaN, NaN, NaN, NaN]);
        }
    }
    let plot1Tep, plot1TepMin, plot1TepMax, plot2Tep, plot2TepMin, plot2TepMax;
    for(let i=0; i<jcApp.nTempNodes; i++) {
        [plot1Tep, plot1TepMin, plot1TepMax] = getPlot1TepAndCi(i);
        [plot2Tep, plot2TepMin, plot2TepMax] = getPlot2TepAndCi(i);
        plot1Tep *= yScale;
        plot1TepMin *= yScale;
        plot1TepMax *= yScale;
        plot2Tep *= yScale;
        plot2TepMin *= yScale;
        plot2TepMax *= yScale;
        if(!jcApp.showPlot1) {
            plot1Tep = NaN;
        }
        if(!jcApp.showPlot2) {
            plot2Tep = NaN;
        }

        data.addRow([jcApp.tempArray[i], NaN, 
            plot1Tep, plot1TepMin, plot1TepMax,
            plot2Tep, plot2TepMin, plot2TepMax]);
    }

    let options = {
      seriesType: 'line',
      series: {0: {type: 'scatter'}},
      title: yLabel,
      titleTextStyle: {bold: true, fontSize: 20,},
      hAxis: {title: xLabel, titleTextStyle: {italic: false, fontSize: 15,},},
      vAxis: {title: yLabel, titleTextStyle: {italic: false, fontSize: 15,},},
      legend: { position: 'bottom', alignment: 'center' },
      intervals: { style: 'area' },
      colors: [jcApp.colorRawData, jcApp.colorPlot1, jcApp.colorPlot2],
      height: jcApp.chartHeight,
      width: jcApp.chartWidth,
    };
  
    chart.draw(data, options);
}
jcApp.getTepData = function(rawdataName) {
    const selectedDataName = jcApp.select.value;
    let tempData = null;
    let tepData = null;
    if (selectedDataName) {
        tempData = jcApp.rawdata[selectedDataName]["temperature [degC]"];
        tepData = jcApp.rawdata[selectedDataName][rawdataName];    
    }
    return [tempData, tepData];
}
jcApp.getElecResiData = function() {
    return jcApp.getTepData("electrical_resistivity [Ohm m]");
}
jcApp.getSeebeckData = function() {
    return jcApp.getTepData("Seebeck_coefficient [V/K]");
}
jcApp.getThrmCondData = function() {
    return jcApp.getTepData("thermal_conductivity [W/m/K]");
}
jcApp.drawElecResiChart = function() {
    function getFuncTepAndCiForPlot(plotNum) {
        const func = function getPlotTepAndCi(i) {
            let tep, std, showCi;
            if (plotNum == 1) {
                tep = jcApp.plot1ElecResiArray[i];
                std = jcApp.plot1ElecResiStdArray[i];
                showCi = jcApp.showPlot1TepCi;
            } else if (plotNum == 2) {
                tep = jcApp.plot2ElecResiArray[i];
                std = jcApp.plot2ElecResiStdArray[i];
                showCi = jcApp.showPlot2TepCi;
            } else {
                throw new Error("Invalid Plot Number!");
            }
            let tepMin = tep - 1.96*std;
            let tepMax = tep + 1.96*std;
            // Do not draw CI if user wanted, or positivity is violated
            if((!showCi) || (tepMin < 0)) {
                tepMin = NaN;
                tepMax = NaN;
            }
            return [tep, tepMin, tepMax];    
        };
        return func;
    };

    jcApp.drawTepChart(jcApp.chartElecResi, "Electrical resistivity (mΩ cm)", 1e5, jcApp.getElecResiData, getFuncTepAndCiForPlot(1), getFuncTepAndCiForPlot(2));
};
jcApp.drawSeebeckChart = function() {
    function getFuncTepAndCiForPlot(plotNum) {
        const func = function getPlotTepAndCi(i) {
            let tep, std, showCi;
            if (plotNum == 1) {
                tep = jcApp.plot1SeebeckArray[i];
                std = jcApp.plot1SeebeckStdArray[i];
                showCi = jcApp.showPlot1TepCi;
            } else if (plotNum == 2) {
                tep = jcApp.plot2SeebeckArray[i];
                std = jcApp.plot2SeebeckStdArray[i];
                showCi = jcApp.showPlot2TepCi;
            } else {
                throw new Error("Invalid Plot Number!");
            }
            let tepMin = tep - 1.96*std;
            let tepMax = tep + 1.96*std;
            // Do not draw CI if user wanted
            if(!showCi) {
                tepMin = NaN;
                tepMax = NaN;
            }
            return [tep, tepMin, tepMax];    
        };
        return func;
    };
    
    jcApp.drawTepChart(jcApp.chartSeebeck, "Seebeck coefficient (μV/K)", 1e6, jcApp.getSeebeckData, getFuncTepAndCiForPlot(1), getFuncTepAndCiForPlot(2));
};
jcApp.drawThrmCondChart = function() {
    function getFuncTepAndCiForPlot(plotNum) {
        const func = function getPlotTepAndCi(i) {
            let tep, std, showCi;
            if (plotNum == 1) {
                tep = jcApp.plot1ThrmCondArray[i];
                std = jcApp.plot1ThrmCondStdArray[i];
                showCi = jcApp.showPlot1TepCi;
            } else if (plotNum == 2) {
                tep = jcApp.plot2ThrmCondArray[i];
                std = jcApp.plot2ThrmCondStdArray[i];
                showCi = jcApp.showPlot2TepCi;
            } else {
                throw new Error("Invalid Plot Number!");
            }
            let tepMin = tep - 1.96*std;
            let tepMax = tep + 1.96*std;
            // Do not draw CI if user wanted or positivity is violated
            if((!showCi) || (tepMin < 0)) {
                tepMin = NaN;
                tepMax = NaN;
            }
            return [tep, tepMin, tepMax];    
        };
        return func;
    };

    jcApp.drawTepChart(jcApp.chartThrmCond, "Thermal conductivity (W/m/K)", 1, jcApp.getThrmCondData, getFuncTepAndCiForPlot(1), getFuncTepAndCiForPlot(2));
};
jcApp.drawElecCondChart = function() {
    function getElecCondData() {
        const [tempData, elecResiData] = jcApp.getElecResiData();
        const elecCondData = elecResiData.map((x) => 1/x);
        return [tempData, elecCondData];
    };
    function getFuncTepAndCiForPlot(plotNum) {
        const func = function getPlotTepAndCi(i) {
            let elecResiTep, elecResiStd, showCi;
            if (plotNum == 1) {
                elecResiTep = jcApp.plot1ElecResiArray[i];
                elecResiStd = jcApp.plot1ElecResiStdArray[i];
                showCi = jcApp.showPlot1TepCi;
            } else if (plotNum == 2) {
                elecResiTep = jcApp.plot2ElecResiArray[i];
                elecResiStd = jcApp.plot2ElecResiStdArray[i];
                showCi = jcApp.showPlot2TepCi;
            } else {
                throw new Error("Invalid Plot Number!");
            }
            let elecResiMin = elecResiTep - 1.96*elecResiStd;
            let elecResiMax = elecResiTep + 1.96*elecResiStd;
            // Do not draw CI if user wanted, or positivity is violated
            if((!showCi) || (elecResiMin < 0)) {
                elecResiMin = NaN;
                elecResiMax = NaN;
            }
            return [1/elecResiTep, 1/elecResiMax, 1/elecResiMin];
        };
        return func;
    };

    jcApp.drawTepChart(jcApp.chartElecCond, "Electrical conductivity (S/cm)", 1e-2, getElecCondData, getFuncTepAndCiForPlot(1), getFuncTepAndCiForPlot(2));
};
jcApp.drawPowerFactorChart = function() {
    function getPowerFactorData() {
        const [tempData, elecResiData] = jcApp.getElecResiData();
        const [, seebeckData] = jcApp.getSeebeckData();
        const powerFactorData = seebeckData.map(function(seebeck, i) {
            return seebeck*seebeck / elecResiData[i];
        });
        return [tempData, powerFactorData];
    };
    function getFuncTepAndCiForPlot(plotNum) {
        const func = function getPlotTepAndCi(i) {
            let elecResiTep, elecResiStd, seebeckTep, seebeckStd, showCi;
            if (plotNum == 1) {
                elecResiTep = jcApp.plot1ElecResiArray[i];
                elecResiStd = jcApp.plot1ElecResiStdArray[i];
                seebeckTep = jcApp.plot1SeebeckArray[i];
                seebeckStd = jcApp.plot1SeebeckStdArray[i];        
                showCi = jcApp.showPlot1zTCi;
            } else if (plotNum == 2) {
                elecResiTep = jcApp.plot2ElecResiArray[i];
                elecResiStd = jcApp.plot2ElecResiStdArray[i];
                seebeckTep = jcApp.plot2SeebeckArray[i];
                seebeckStd = jcApp.plot2SeebeckStdArray[i];        
                showCi = jcApp.showPlot2zTCi;
            } else {
                throw new Error("Invalid Plot Number!");
            }
            const elecResiMin = elecResiTep - 1.96*elecResiStd;
            const elecResiMax = elecResiTep + 1.96*elecResiStd;
            const seebeckMin = seebeckTep - 1.96*seebeckStd;
            const seebeckMax = seebeckTep + 1.96*seebeckStd;    
            const seebeckSquared = Math.pow(seebeckTep, 2);
            let seebeckSquaredMin = Math.min(seebeckSquared, Math.pow(seebeckMin, 2), Math.pow(seebeckMax, 2));
            let seebeckSquaredMax = Math.max(seebeckSquared, Math.pow(seebeckMin, 2), Math.pow(seebeckMax, 2));
            // the above min/max fails when the sign of Seebeck coefficient changes
            if ((seebeckMin < 0) && (seebeckMax > 0)) {
                seebeckSquaredMin = 0;
            }
            // Do not draw CI if user wanted
            if (!showCi) {
                seebeckSquaredMin = NaN;
                seebeckSquaredMax = NaN;
            }
            return [seebeckSquared/elecResiTep, seebeckSquaredMin/elecResiMax, seebeckSquaredMax/elecResiMin];
        };
        return func;
    };

    jcApp.drawTepChart(jcApp.chartPowerFactor, "Power factor (mW/m/K\u00B2)", 1e3, getPowerFactorData, getFuncTepAndCiForPlot(1), getFuncTepAndCiForPlot(2));
};
jcApp.drawFigureOfMeritChart = function() {
    function getFigureOfMeritData() {
        const [tempData, elecResiData] = jcApp.getElecResiData();
        const [, seebeckData] = jcApp.getSeebeckData();
        const [, thrmCondData] = jcApp.getThrmCondData();
        const figureOfMeritData = seebeckData.map(function(seebeck, i) {
            return seebeck*seebeck / (elecResiData[i]*thrmCondData[i]) * (tempData[i] + 273.15); // absolute temperature (K)
        });
        return [tempData, figureOfMeritData];
    };
    function getFuncTepAndCiForPlot(plotNum) {
        const func = function getPlotTepAndCi(i) {
            let elecResiTep, elecResiStd, seebeckTep, seebeckStd, thrmCondTep, thrmCondStd, showCi;
            if (plotNum == 1) {
                elecResiTep = jcApp.plot1ElecResiArray[i];
                elecResiStd = jcApp.plot1ElecResiStdArray[i];
                seebeckTep = jcApp.plot1SeebeckArray[i];
                seebeckStd = jcApp.plot1SeebeckStdArray[i];
                thrmCondTep = jcApp.plot1ThrmCondArray[i];
                thrmCondStd = jcApp.plot1ThrmCondStdArray[i];    
                showCi = jcApp.showPlot1zTCi;
            } else if (plotNum == 2) {
                elecResiTep = jcApp.plot2ElecResiArray[i];
                elecResiStd = jcApp.plot2ElecResiStdArray[i];
                seebeckTep = jcApp.plot2SeebeckArray[i];
                seebeckStd = jcApp.plot2SeebeckStdArray[i];        
                thrmCondTep = jcApp.plot2ThrmCondArray[i];
                thrmCondStd = jcApp.plot2ThrmCondStdArray[i];    
                showCi = jcApp.showPlot2zTCi;
            } else {
                throw new Error("Invalid Plot Number!");
            }
            const absTemp = jcApp.tempArray[i] + 273.15;  // absolute temperature (K)
            const elecResiMin = elecResiTep - 1.96*elecResiStd;
            const elecResiMax = elecResiTep + 1.96*elecResiStd;
            const seebeckMin = seebeckTep - 1.96*seebeckStd;
            const seebeckMax = seebeckTep + 1.96*seebeckStd;    
            const thrmCondMin = thrmCondTep - 1.96*thrmCondStd;
            const thrmCondMax = thrmCondTep + 1.96*thrmCondStd;    
            const seebeckSquared = Math.pow(seebeckTep, 2);
            let seebeckSquaredMin = Math.min(seebeckSquared, Math.pow(seebeckMin, 2), Math.pow(seebeckMax, 2));
            let seebeckSquaredMax = Math.max(seebeckSquared, Math.pow(seebeckMin, 2), Math.pow(seebeckMax, 2));
            // the above min/max fails when the sign of Seebeck coefficient changes
            if ((seebeckMin < 0) && (seebeckMax > 0)) {
                seebeckSquaredMin = 0;
            }
            // Do not draw CI if user wanted
            if (!showCi) {
                seebeckSquaredMin = NaN;
                seebeckSquaredMax = NaN;
            }
            return [seebeckSquared/(elecResiTep*thrmCondTep)*absTemp,
                seebeckSquaredMin/(elecResiMax*thrmCondMax)*absTemp,
                seebeckSquaredMax/(elecResiMin*thrmCondMin)*absTemp];
        }
        return func;
    };

    jcApp.drawTepChart(jcApp.chartFigureOfMerit, "Figure of merit zT (1)", 1, getFigureOfMeritData, getFuncTepAndCiForPlot(1), getFuncTepAndCiForPlot(2));
};


jcApp.initMeanLann = function() {
    const jsonObj = jcApp.jsonObjBiTeMeanLann;
    const embeddingNet = new FullyConnectedNeuralNetwork(5,
        jsonObj["embeddingNet"]["weightsArray"],
        jsonObj["embeddingNet"]["biasesArray"],
        jsonObj["embeddingNet"]["activationArray"]
    );
    const meanDictionaryNet = new FullyConnectedNeuralNetwork(4,
        jsonObj["dictionaryNet"]["weightsArray"],
        jsonObj["dictionaryNet"]["biasesArray"],
        jsonObj["dictionaryNet"]["activationArray"]
    );
    jcApp.meanLann = new BiTeMeanLann(embeddingNet, meanDictionaryNet);
}
jcApp.initLann = function() {
    let jsonObj = jcApp.jsonObjBiTeLann;
    let embeddingNet = new FullyConnectedNeuralNetwork(5,
        jsonObj["embeddingNet"]["weightsArray"],
        jsonObj["embeddingNet"]["biasesArray"],
        jsonObj["embeddingNet"]["activationArray"]
    );
    let meanDictionaryNet = new FullyConnectedNeuralNetwork(4,
        jsonObj["meanDictionaryNet"]["weightsArray"],
        jsonObj["meanDictionaryNet"]["biasesArray"],
        jsonObj["meanDictionaryNet"]["activationArray"]
    );
    let stdDictionaryNet = new FullyConnectedNeuralNetwork(4,
        jsonObj["stdDictionaryNet"]["weightsArray"],
        jsonObj["stdDictionaryNet"]["biasesArray"],
        jsonObj["stdDictionaryNet"]["activationArray"]
    );
    jcApp.lann = new BiTeLann(embeddingNet, meanDictionaryNet, stdDictionaryNet);
    console.log("Machine learning model initialized.");
};

jcApp.getLinearSpace = function(x0, xf, numNodes) {
    const vec = new Float64Array(numNodes);
    const dx = (xf-x0)/(numNodes-1);
    for(let i=0; i<vec.length; i++) {
        vec[i] = (x0 + dx*i);
    };
    vec[vec.length-1] = xf;

    return vec;
};

jcApp.jsonObjBiTeMeanLann = {"embeddingNet": {"weightsArray": [[0.3936586776872572, -0.23473166882298177, -0.3753017563379596, 0.011286703108257076, -0.10818528853446602, -0.6502619644816681, -0.10751356579955935, 0.47721445694003467, 0.10610241743914647, -0.398328599340446, 0.0016696245949437993, 0.36679137080076374, -0.7371478686982197, -0.2200378933046389, 0.3563029572967085, 0.2846979027151496, -0.7575454251143153, 0.46038040235675504, -0.491030543441278, 0.40271021271660157, -0.5431950813495217, 0.4464540603664178, -0.6109365395701555, 0.40865194281753614, 0.40630238654073797, -0.030807858599031315, -0.6379360976202108, 0.31566394871612713, 0.31062878995801774, -0.049228077276660794, 0.31456874210389196, -0.13945244042326915, -0.2841542790067249, -0.08637082048067668, 0.052600322650213235, 0.4817896607806807, -0.08940789901467056, 0.4499412936054613, -0.029628642095148844, 0.2358566496200143, -0.8574302104318285, 0.6155293638156466, 0.3672226862680059, 0.17004129834707696, -0.16551168116546394, 1.0229672275121264, -0.14145743219443024, 0.21766122201681476, -0.20497833816502906, -0.4277369642750238, -0.5125244313302578, 0.029894380860772885, 0.584293740633192, -0.09594031595223484, 0.376087944529117, -0.5744227083320161, 0.01683644562164975, 0.5297494741601925, 0.11186870047147905, 0.24356940919627273, 0.8103946853296118, -0.20808581581194863, 0.16994240559569782, 0.22882789570709483, -0.429493324369064, 0.6805310727163958, 0.5049590429288489, 0.2288137129628534, -0.2619772008067191, -0.09386666828326116, 0.13135471852479305, 0.08036245139828521, 0.16457488006719737, -0.3790348266861358, 0.20314581095527443, -0.9477568763939715, -0.0659195334011569, 0.44783986583079916, 0.375064326466662, 0.21019007935496392], [0.3653968299156001, 0.1004754571022186, 0.1809304672346645, 0.5376052522837435, -0.03877387599390508, 0.0898432042157332, 0.28446877915965724, -0.049859684284259555, -0.23843523323875318, 0.05779046384187468, 0.23311169072452106, -0.23812127282870293, -0.08692672929207859, -0.13975487046801696, -0.2634046069321904, -0.47160128573401183, 0.2353817833420021, -0.15079666455003948, -0.1171854650270034, 0.12120142343183275, 0.04064844001194977, 0.4594540467713153, -0.27974980631192564, -0.31074727848388395, -0.3343658835244429, -0.036958250955620144, 0.4189903592465155, 0.15939886349377919, -0.13977316039386012, -0.14159315934445177, 0.2048055799781964, -0.4080917066908995, -0.2550474668090807, -0.04369161627218415, -0.08357519538173361, 0.0820732361819033, -0.40248071113348766, -0.38556199871128816, 0.4007423303449824, -0.1398167512997594, -0.1244113843826407, 0.07043828923889743, 0.3886371210536565, -0.22155405444093879, 0.27005856697075187, -0.4862061291663474, 0.3074671068718362, -0.4738722300123639, -0.28420463956961367, 0.37140292338088443, 0.296655043471607, -0.4401822960513517, 0.39010907221118885, 0.30136545108405666, -0.2031384460582524, -0.1884468382505592, -0.05723336236985058, -0.7232433543268793, 0.008526094583639947, 0.45013743217703245, 0.22996977976230906, -0.44576808013294905, -0.29619565781011514, 0.4604393013863395, -0.06758100504496595, -0.5664921793614394, -0.2071957286074108, -0.15278779685158433, -0.28422878247930033, -0.2783952895257113, -0.04128929727196191, 0.4388608975372502, -0.13883322677364177, 0.39048461678831564, 0.19241345736006266, -0.5795062827252003, -0.01833322917099954, 0.20809640620760173, -0.008109391360037618, -0.11455425258198768, 0.4730259631361817, 0.03873208563307776, -0.6242018719700332, 0.14107194577993878, -0.3076634196569107, 0.1426208267830692, -0.061527023206996496, 0.2474858041471625, -0.09225056691173258, -0.13163343082298884, 0.15586755650201783, -0.34775608328958063, 0.49911935183923545, -0.3724389140780479, -0.1990389220181213, 0.35098264358810477, -0.13977753819965244, -0.39442239265405193, 0.2988220277779315, -0.13915097478955843, -0.014696550745138015, -0.39674011968294715, 0.19468785781230075, -0.32552300787234223, 0.16496560406499125, 0.05804051303627177, -0.20060001331621835, -0.4400499753751459, 0.6859657375815527, -0.07798554189398728, 0.17027179001483167, -0.28986086253655013, -0.18905914323843054, 0.022891669718104773, -0.6294209592295424, 0.19281447543864338, -0.5600791849874366, 0.48337910715381693, -0.09414186336629865, 0.2000494785186866, -0.10505772356690958, 0.6300823533423928, -0.28156183943310875, -0.2694484950297106, 0.3909054531953571, 0.11014741287785539, 0.08130738812516354, 0.19548279639059563, -0.07408267965788841, -0.3772242321736844, -0.12959298191380828, 0.25497821573142126, 0.11773228001325811, -0.022589670834182, -0.29054788734196657, -0.37889678669985705, -0.33927148431646775, 0.4071447098422936, -0.289972732565895, -0.11688174636364262, 0.24406595026785413, 0.008804380183305413, 0.3031557683606157, 0.32882487316316, -0.20496644115674006, -0.2541686367127316, 0.2791281856922178, -0.12044954178010416, 0.4316731703234361, -0.08922431364574294, 0.07537888610147792, -0.5904055865421195, 0.4967208235577406, -1.366456780030379, 0.21868679250900847, 0.22251400382824418, -0.8067650555195416, -0.5372750678561942, 0.14852118208054205, 1.163755901727595, 0.19704759855558002, -0.47299480182885506, -0.060319817068428115, -0.09622185716932331, 0.3215157504305932, -0.3500933214918878, -1.9028245398769754e-05, -0.12056007632962525, -0.523641721118721, 0.42759335223058154, -0.4630304567806558, -0.48143089793445387, 0.12445687588986364, 0.17526765015182957, 0.08942256786461737, -0.7403861738592863, -0.09013189302928272, 0.36775592631640275, -0.18393473652732714, -0.4920717823002023, 0.3293219321783135, -0.27851691935127404, -0.27540902562649766, -0.2503647033440489, 0.6322453164845951, -0.10358125723549817, 0.04231311410707451, -0.2439926719717727, -0.5934271767398767, 0.08042649804193662, 0.14052871037149842, 0.1392699667613604, 0.049122940152847794, -0.22256993868533245, 0.3413122445290882, -0.03055951327995345, 0.3779266541282942, -0.34160958399652935, 0.23371369899920671, 0.10744490064038746, -0.21735509937952258, 0.2650860022223043, -0.32066549668660643, -0.6094330233535095, 0.3772989564352911, 0.18064904995841582, -0.22224243178694447, -0.7396990224354556, -0.2919942892216123, 0.1436172269804917, -0.2121945800249118, 0.06652258283769787, -0.4927330896788201, 0.16389094539812304, -0.31768471145719357, 0.10905214217841819, -0.4151198219222369, -0.09473938283571696, -0.09834832153656677, -0.33987149297196106, 0.31098631045418823, 0.1148797817583156, 0.3886328219444132, 0.10351823754153362, 0.41390896015291134, -0.6605709418825559, 0.03503134638818437, -0.21436056628234867, 0.11993784460693874, -0.47021886530410956, -0.1793790413281834, -0.12921045415623045, -0.6204907486967539, 0.6298267360402, -0.411376069609189, -1.1328806468093378, 0.1738298476321308, 0.2905153759753984, 0.09458865584202515, -1.5999588909559708, -0.12747265507074523, -0.01704504242079146, -0.1791353350190065, 0.2591343637250338, -0.10737658393979028, 0.20770788920562455, 0.13934452179982781, 0.14347182021942195, -0.37058845734999535, 0.0162729279966972, 0.057071756741298704, 0.5369380531049249, 0.1916850498626315, -0.08541410975688242, 0.2622838289398126, 0.3147317496876883], [0.2589420318297505, 0.27742336932523237, 0.3560566759416356, 0.1525275231063591, -0.10166723226171026, 0.18927386805169796, -0.548531558211767, 0.6004720602077637, 0.04028180999504379, 0.15340197926094384, -0.5580384825280641, -0.4951680142051368, -0.29806832835484437, 0.4688260350236564, -0.44668602735563656, 0.04552124453675927, 0.38270018490091645, 0.20506881573280256, 0.5018204255235241, -0.7814984219710244, 0.2840104502366258, -0.2760496132957357, -0.11565378259420891, -0.004687218502290189, 0.10408081863471022, -1.4401227166816823, -0.4883516322625387, -0.5696782912070293, 0.22883210950699687, 0.4717211491722532, 0.01439528740672571, 0.059310664756895046, 0.44456303410410797, 0.23558247665194446, 0.5398484466427781, 0.005097274025058722, -0.1094371495247834, -0.29784449579605154, 0.15107944455674033, 0.17117260100057793, -0.010526961423631753, 0.15444595219480864, 0.801819311964486, -0.08828972416860086, 0.5119860408422466, 0.25429865572155075, 0.6834114084317254, -0.5802821635801849]], "biasesArray": [[0.06940879499879044, 0.18834966094688918, -0.02649649398510797, -0.09514193422242147, 0.020836582274290985, -0.00935722582808401, 0.05301085706484741, -0.21380347992187765, 0.09251944313144787, 0.037494547358072026, 0.0006213286265637448, 0.1301882656950864, 0.006319450056927245, -0.07679202467483941, -0.1212104702771394, 0.1968533421517101], [-0.19147948882497665, -0.004683314643751354, -0.13036181530687388, 0.03514766804196115, 0.012012041067885157, -0.06809824306072008, 0.016788265037509815, 0.04318899889739899, -0.047361571582128184, -0.05915128811721764, -0.04588367926977966, 0.0703523311751324, -0.062194791421392344, 0.004186862476848579, 0.017291225539668605, -0.02127099416917539], [0.009608791223286325, -0.09332725705583443, -0.012901578289758565]], "activationArray": ["elu", "elu", "linear"]}, "dictionaryNet": {"weightsArray": [[0.5206704768214222, -0.016744190632433043, 0.32184830396478264, -1.636850556621218, 0.37552402288289455, 0.46474911334335284, 0.02184916026993797, -0.4401640606644141, 0.2323455634205349, 0.3432637312191049, -0.43214333635467755, -0.05299413594777176, -0.7432565354568171, 0.7981043456908167, 1.3256904785230579, -1.1061997351494155, -0.6308328871527152, 0.0007724229896406811, -0.18011865666282467, 1.1984965275385349, -0.4430203730079615, -0.7080054184647092, 0.26510738520762017, -2.0807386940015244, -0.2023908135729908, -0.02742172597435756, 0.8736321593945315, 1.1383004031907058, -0.5710680369148811, 0.6697666408601071, -0.35096401239036684, 0.6169037626231156], [-0.37281070125101157, 0.5429536576448634, 0.07884421444760795, 0.08627477133848255, -0.07006233041305253, 0.256819625134199, 0.175351171765259, 0.17415318341021616, -0.3673743181499673, 0.41727273989942665, -0.07770405597625904, 0.18925590008519158, 0.05189227254103819, 0.7464394620103841, 0.6204434034289213, 0.3058628386564285, -0.5100995007582616, 0.3655395567260059, 0.23349900450632735, -0.12412856757953208, -0.6644342799981033, 0.09575225652331443, 0.32672399390679824, -0.5064390941229002, -0.11132047106197077, 0.2335388422878275, 0.6414823684580147, 0.4071078385659196, 0.2838191180509349, -0.6194113358012177, 0.3741743809455516, 0.4952272429869598, 0.043210939742756714, -0.2824093212636338, -0.11945191271757467, 0.29797035850776865, -0.4013130821333212, 0.342390482749825, -0.4457481675328885, -0.5041202641020807, -0.17236603268722, 0.3647008880174394, 0.20388786298611872, 0.17668432013153276, -0.13424252460437072, -1.5836689876173093, 0.7066395581743884, 0.6626871298264875, -0.7947882121759304, 0.4148524656114012, 0.33675879633589767, 0.21819738223923835, 0.020232615448086855, -0.4275347133491682, 0.5185917905368974, -0.16540354664504892, -0.5458676379761771, 0.1720387708867441, -0.02243405247007742, 0.3728727792685184, -0.7907652656898329, 0.18573438858861965, -0.2504839285324757, 0.18155169315221092], [0.6403285711320076, 0.1586420637876473, 0.15726240203345448, 0.9724747430935425, -0.5915439020382997, -0.9693782341732604, 0.8131568626807855, -0.47616736781885016, 0.3076954917362582, 0.5719025098723899, -1.0129454065576886, -0.1627212099893545, -0.4321856738293865, -0.4038882526295665, -0.31341132569099234, 0.04408253994407846, -0.3618657136720517, -0.23373169437459893, -0.7930999889802066, 0.13614593804862146, 0.7854109270626382, 0.19691050308196226, 0.05693639531866658, -0.863468388023132]], "biasesArray": [[0.3414152190391911, -0.08622229860961085, -0.030377797316199228, -0.45764344746742897, -0.07393732253125723, 0.1285816350860901, 0.12583279902372457, 0.152048237865095], [0.09859145750718555, -0.006814916370477385, -0.013462201698544238, -0.033056289029914636, -0.035252705734336594, -0.14481142057967256, 0.22962613455425543, -0.5387544246803784], [0.19987462278144963, 0.036746421518248536, 0.6264658493747752]], "activationArray": ["elu", "elu", "linear"]}};

jcApp.jsonObjBiTeLann = {"embeddingNet": {"weightsArray": [[0.3936586776872572, -0.23473166882298177, -0.3753017563379596, 0.011286703108257076, -0.10818528853446602, -0.6502619644816681, -0.10751356579955935, 0.47721445694003467, 0.10610241743914647, -0.398328599340446, 0.0016696245949437993, 0.36679137080076374, -0.7371478686982197, -0.2200378933046389, 0.3563029572967085, 0.2846979027151496, -0.7575454251143153, 0.46038040235675504, -0.491030543441278, 0.40271021271660157, -0.5431950813495217, 0.4464540603664178, -0.6109365395701555, 0.40865194281753614, 0.40630238654073797, -0.030807858599031315, -0.6379360976202108, 0.31566394871612713, 0.31062878995801774, -0.049228077276660794, 0.31456874210389196, -0.13945244042326915, -0.2841542790067249, -0.08637082048067668, 0.052600322650213235, 0.4817896607806807, -0.08940789901467056, 0.4499412936054613, -0.029628642095148844, 0.2358566496200143, -0.8574302104318285, 0.6155293638156466, 0.3672226862680059, 0.17004129834707696, -0.16551168116546394, 1.0229672275121264, -0.14145743219443024, 0.21766122201681476, -0.20497833816502906, -0.4277369642750238, -0.5125244313302578, 0.029894380860772885, 0.584293740633192, -0.09594031595223484, 0.376087944529117, -0.5744227083320161, 0.01683644562164975, 0.5297494741601925, 0.11186870047147905, 0.24356940919627273, 0.8103946853296118, -0.20808581581194863, 0.16994240559569782, 0.22882789570709483, -0.429493324369064, 0.6805310727163958, 0.5049590429288489, 0.2288137129628534, -0.2619772008067191, -0.09386666828326116, 0.13135471852479305, 0.08036245139828521, 0.16457488006719737, -0.3790348266861358, 0.20314581095527443, -0.9477568763939715, -0.0659195334011569, 0.44783986583079916, 0.375064326466662, 0.21019007935496392], [0.3653968299156001, 0.1004754571022186, 0.1809304672346645, 0.5376052522837435, -0.03877387599390508, 0.0898432042157332, 0.28446877915965724, -0.049859684284259555, -0.23843523323875318, 0.05779046384187468, 0.23311169072452106, -0.23812127282870293, -0.08692672929207859, -0.13975487046801696, -0.2634046069321904, -0.47160128573401183, 0.2353817833420021, -0.15079666455003948, -0.1171854650270034, 0.12120142343183275, 0.04064844001194977, 0.4594540467713153, -0.27974980631192564, -0.31074727848388395, -0.3343658835244429, -0.036958250955620144, 0.4189903592465155, 0.15939886349377919, -0.13977316039386012, -0.14159315934445177, 0.2048055799781964, -0.4080917066908995, -0.2550474668090807, -0.04369161627218415, -0.08357519538173361, 0.0820732361819033, -0.40248071113348766, -0.38556199871128816, 0.4007423303449824, -0.1398167512997594, -0.1244113843826407, 0.07043828923889743, 0.3886371210536565, -0.22155405444093879, 0.27005856697075187, -0.4862061291663474, 0.3074671068718362, -0.4738722300123639, -0.28420463956961367, 0.37140292338088443, 0.296655043471607, -0.4401822960513517, 0.39010907221118885, 0.30136545108405666, -0.2031384460582524, -0.1884468382505592, -0.05723336236985058, -0.7232433543268793, 0.008526094583639947, 0.45013743217703245, 0.22996977976230906, -0.44576808013294905, -0.29619565781011514, 0.4604393013863395, -0.06758100504496595, -0.5664921793614394, -0.2071957286074108, -0.15278779685158433, -0.28422878247930033, -0.2783952895257113, -0.04128929727196191, 0.4388608975372502, -0.13883322677364177, 0.39048461678831564, 0.19241345736006266, -0.5795062827252003, -0.01833322917099954, 0.20809640620760173, -0.008109391360037618, -0.11455425258198768, 0.4730259631361817, 0.03873208563307776, -0.6242018719700332, 0.14107194577993878, -0.3076634196569107, 0.1426208267830692, -0.061527023206996496, 0.2474858041471625, -0.09225056691173258, -0.13163343082298884, 0.15586755650201783, -0.34775608328958063, 0.49911935183923545, -0.3724389140780479, -0.1990389220181213, 0.35098264358810477, -0.13977753819965244, -0.39442239265405193, 0.2988220277779315, -0.13915097478955843, -0.014696550745138015, -0.39674011968294715, 0.19468785781230075, -0.32552300787234223, 0.16496560406499125, 0.05804051303627177, -0.20060001331621835, -0.4400499753751459, 0.6859657375815527, -0.07798554189398728, 0.17027179001483167, -0.28986086253655013, -0.18905914323843054, 0.022891669718104773, -0.6294209592295424, 0.19281447543864338, -0.5600791849874366, 0.48337910715381693, -0.09414186336629865, 0.2000494785186866, -0.10505772356690958, 0.6300823533423928, -0.28156183943310875, -0.2694484950297106, 0.3909054531953571, 0.11014741287785539, 0.08130738812516354, 0.19548279639059563, -0.07408267965788841, -0.3772242321736844, -0.12959298191380828, 0.25497821573142126, 0.11773228001325811, -0.022589670834182, -0.29054788734196657, -0.37889678669985705, -0.33927148431646775, 0.4071447098422936, -0.289972732565895, -0.11688174636364262, 0.24406595026785413, 0.008804380183305413, 0.3031557683606157, 0.32882487316316, -0.20496644115674006, -0.2541686367127316, 0.2791281856922178, -0.12044954178010416, 0.4316731703234361, -0.08922431364574294, 0.07537888610147792, -0.5904055865421195, 0.4967208235577406, -1.366456780030379, 0.21868679250900847, 0.22251400382824418, -0.8067650555195416, -0.5372750678561942, 0.14852118208054205, 1.163755901727595, 0.19704759855558002, -0.47299480182885506, -0.060319817068428115, -0.09622185716932331, 0.3215157504305932, -0.3500933214918878, -1.9028245398769754e-05, -0.12056007632962525, -0.523641721118721, 0.42759335223058154, -0.4630304567806558, -0.48143089793445387, 0.12445687588986364, 0.17526765015182957, 0.08942256786461737, -0.7403861738592863, -0.09013189302928272, 0.36775592631640275, -0.18393473652732714, -0.4920717823002023, 0.3293219321783135, -0.27851691935127404, -0.27540902562649766, -0.2503647033440489, 0.6322453164845951, -0.10358125723549817, 0.04231311410707451, -0.2439926719717727, -0.5934271767398767, 0.08042649804193662, 0.14052871037149842, 0.1392699667613604, 0.049122940152847794, -0.22256993868533245, 0.3413122445290882, -0.03055951327995345, 0.3779266541282942, -0.34160958399652935, 0.23371369899920671, 0.10744490064038746, -0.21735509937952258, 0.2650860022223043, -0.32066549668660643, -0.6094330233535095, 0.3772989564352911, 0.18064904995841582, -0.22224243178694447, -0.7396990224354556, -0.2919942892216123, 0.1436172269804917, -0.2121945800249118, 0.06652258283769787, -0.4927330896788201, 0.16389094539812304, -0.31768471145719357, 0.10905214217841819, -0.4151198219222369, -0.09473938283571696, -0.09834832153656677, -0.33987149297196106, 0.31098631045418823, 0.1148797817583156, 0.3886328219444132, 0.10351823754153362, 0.41390896015291134, -0.6605709418825559, 0.03503134638818437, -0.21436056628234867, 0.11993784460693874, -0.47021886530410956, -0.1793790413281834, -0.12921045415623045, -0.6204907486967539, 0.6298267360402, -0.411376069609189, -1.1328806468093378, 0.1738298476321308, 0.2905153759753984, 0.09458865584202515, -1.5999588909559708, -0.12747265507074523, -0.01704504242079146, -0.1791353350190065, 0.2591343637250338, -0.10737658393979028, 0.20770788920562455, 0.13934452179982781, 0.14347182021942195, -0.37058845734999535, 0.0162729279966972, 0.057071756741298704, 0.5369380531049249, 0.1916850498626315, -0.08541410975688242, 0.2622838289398126, 0.3147317496876883], [0.2589420318297505, 0.27742336932523237, 0.3560566759416356, 0.1525275231063591, -0.10166723226171026, 0.18927386805169796, -0.548531558211767, 0.6004720602077637, 0.04028180999504379, 0.15340197926094384, -0.5580384825280641, -0.4951680142051368, -0.29806832835484437, 0.4688260350236564, -0.44668602735563656, 0.04552124453675927, 0.38270018490091645, 0.20506881573280256, 0.5018204255235241, -0.7814984219710244, 0.2840104502366258, -0.2760496132957357, -0.11565378259420891, -0.004687218502290189, 0.10408081863471022, -1.4401227166816823, -0.4883516322625387, -0.5696782912070293, 0.22883210950699687, 0.4717211491722532, 0.01439528740672571, 0.059310664756895046, 0.44456303410410797, 0.23558247665194446, 0.5398484466427781, 0.005097274025058722, -0.1094371495247834, -0.29784449579605154, 0.15107944455674033, 0.17117260100057793, -0.010526961423631753, 0.15444595219480864, 0.801819311964486, -0.08828972416860086, 0.5119860408422466, 0.25429865572155075, 0.6834114084317254, -0.5802821635801849]], "biasesArray": [[0.06940879499879044, 0.18834966094688918, -0.02649649398510797, -0.09514193422242147, 0.020836582274290985, -0.00935722582808401, 0.05301085706484741, -0.21380347992187765, 0.09251944313144787, 0.037494547358072026, 0.0006213286265637448, 0.1301882656950864, 0.006319450056927245, -0.07679202467483941, -0.1212104702771394, 0.1968533421517101], [-0.19147948882497665, -0.004683314643751354, -0.13036181530687388, 0.03514766804196115, 0.012012041067885157, -0.06809824306072008, 0.016788265037509815, 0.04318899889739899, -0.047361571582128184, -0.05915128811721764, -0.04588367926977966, 0.0703523311751324, -0.062194791421392344, 0.004186862476848579, 0.017291225539668605, -0.02127099416917539], [0.009608791223286325, -0.09332725705583443, -0.012901578289758565]], "activationArray": ["elu", "elu", "linear"]}, "meanDictionaryNet": {"weightsArray": [[0.5206704768214222, -0.016744190632433043, 0.32184830396478264, -1.636850556621218, 0.37552402288289455, 0.46474911334335284, 0.02184916026993797, -0.4401640606644141, 0.2323455634205349, 0.3432637312191049, -0.43214333635467755, -0.05299413594777176, -0.7432565354568171, 0.7981043456908167, 1.3256904785230579, -1.1061997351494155, -0.6308328871527152, 0.0007724229896406811, -0.18011865666282467, 1.1984965275385349, -0.4430203730079615, -0.7080054184647092, 0.26510738520762017, -2.0807386940015244, -0.2023908135729908, -0.02742172597435756, 0.8736321593945315, 1.1383004031907058, -0.5710680369148811, 0.6697666408601071, -0.35096401239036684, 0.6169037626231156], [-0.37281070125101157, 0.5429536576448634, 0.07884421444760795, 0.08627477133848255, -0.07006233041305253, 0.256819625134199, 0.175351171765259, 0.17415318341021616, -0.3673743181499673, 0.41727273989942665, -0.07770405597625904, 0.18925590008519158, 0.05189227254103819, 0.7464394620103841, 0.6204434034289213, 0.3058628386564285, -0.5100995007582616, 0.3655395567260059, 0.23349900450632735, -0.12412856757953208, -0.6644342799981033, 0.09575225652331443, 0.32672399390679824, -0.5064390941229002, -0.11132047106197077, 0.2335388422878275, 0.6414823684580147, 0.4071078385659196, 0.2838191180509349, -0.6194113358012177, 0.3741743809455516, 0.4952272429869598, 0.043210939742756714, -0.2824093212636338, -0.11945191271757467, 0.29797035850776865, -0.4013130821333212, 0.342390482749825, -0.4457481675328885, -0.5041202641020807, -0.17236603268722, 0.3647008880174394, 0.20388786298611872, 0.17668432013153276, -0.13424252460437072, -1.5836689876173093, 0.7066395581743884, 0.6626871298264875, -0.7947882121759304, 0.4148524656114012, 0.33675879633589767, 0.21819738223923835, 0.020232615448086855, -0.4275347133491682, 0.5185917905368974, -0.16540354664504892, -0.5458676379761771, 0.1720387708867441, -0.02243405247007742, 0.3728727792685184, -0.7907652656898329, 0.18573438858861965, -0.2504839285324757, 0.18155169315221092], [0.6403285711320076, 0.1586420637876473, 0.15726240203345448, 0.9724747430935425, -0.5915439020382997, -0.9693782341732604, 0.8131568626807855, -0.47616736781885016, 0.3076954917362582, 0.5719025098723899, -1.0129454065576886, -0.1627212099893545, -0.4321856738293865, -0.4038882526295665, -0.31341132569099234, 0.04408253994407846, -0.3618657136720517, -0.23373169437459893, -0.7930999889802066, 0.13614593804862146, 0.7854109270626382, 0.19691050308196226, 0.05693639531866658, -0.863468388023132]], "biasesArray": [[0.3414152190391911, -0.08622229860961085, -0.030377797316199228, -0.45764344746742897, -0.07393732253125723, 0.1285816350860901, 0.12583279902372457, 0.152048237865095], [0.09859145750718555, -0.006814916370477385, -0.013462201698544238, -0.033056289029914636, -0.035252705734336594, -0.14481142057967256, 0.22962613455425543, -0.5387544246803784], [0.19987462278144963, 0.036746421518248536, 0.6264658493747752]], "activationArray": ["elu", "elu", "linear"]}, "stdDictionaryNet": {"weightsArray": [[-0.823641100862015, -0.8568810013644278, 0.3679586385109779, 0.16928821540013622, 0.170730737269674, -0.44635647280026625, -0.48073339468481113, -0.17030574374660257, 0.04558650667741766, -0.05588342381441645, 0.5813366265111989, -0.2560001179103314, 0.04236050962732992, -0.6606351422767468, -0.4434792596069909, -0.42737682438093155, 0.416093406649523, 0.8903782214517701, -0.036227100811933324, 0.38550570859298217, 0.16319448739828515, -0.8610959831826553, 0.371197999861276, 0.5292580554643276, -0.6646928046947891, 0.13815063273825406, -0.04817060751706544, 0.07942717659732039, 0.3918008593348981, 0.25655916634174813, -0.16077824879367286, -0.6309917213146544], [-0.08221256520529502, -0.300971477333716, -0.19377479691335447, -0.329800467845572, 0.03549752260457465, -0.17736281626861317, 0.34023439939564154, -0.11831176389538318, 0.055274317396477896, 0.35726951431004145, -0.43384706475813406, 0.2589135241052325, 0.27200447231937075, 0.14610886349798513, -0.46276406770756645, -0.5445474950520955, -0.5678455280881671, 0.5481255621543025, -0.26904399751697844, -0.48952115577164335, 0.4259922882030248, -0.5830299491714517, 0.5825844949407843, 0.22330067113027793, 0.46357191887819815, 0.3192149003759885, -0.5780422166461016, -0.07734394765195816, 0.619509480118995, 0.10384733807716903, -0.47328077394837514, -0.3618826443011404, 0.03182632846821012, 0.08227053919987551, -0.5927418356251247, -0.4598365504649699, 0.005900790101355012, 0.09667633210754628, -0.32121992846558317, 0.6271995418552814, 0.1488543457577849, -0.02918272217746535, 0.151281004639269, -0.09325715415466515, 0.2826493501222056, -0.23359073901135044, -0.29922186966618863, -0.026112868629645992, 0.255858110470571, 0.28358557438281007, 0.401804435606325, 0.2833617713913529, 0.4718598897029275, 0.21213029053011345, 0.054569955102542346, -0.2900427588400298, 0.012198802010953923, 0.30809559516997487, 0.6024770969839219, -0.5252111926185075, 0.5308139410858037, 0.2997531832004103, 0.33878707731686514, -0.4809137277637499], [0.13833078205774277, 0.2448903984139252, -0.5526967668896849, -0.7066906993671194, 0.15182227273432763, -0.04565232764498657, -0.35972167422099577, -0.10344388716986049, -0.41592646119840754, -0.47115351049912946, 0.3642168069610243, -1.0092076882894605, -0.5735408249618249, 0.073148612278382, 0.03534266433290372, -0.20025121774308158, 0.4599243782453362, -0.340515592497525, -0.6799635172226562, -0.42203066394876687, -0.015738911483709657, 0.5485148331478277, -0.5379850419699213, -0.6987637696601707]], "biasesArray": [[0.29732515942494286, 0.14616515360326238, -0.2716616022201673, -0.08260429394876126, 0.1725999018606355, 0.2264903805741097, -0.1924882413788285, -0.20286027769036788], [0.02108578853913196, 0.13162230456359414, 0.019421103726670583, 0.2587647236181903, 0.15808188770697076, -0.12968803088398545, 0.18197447399236222, 0.15750445846261746], [-0.20143863698122952, -0.24828778136457905, -0.1311335033538169]], "activationArray": ["elu", "elu", "softplus"]}};

jcApp.rawdata = {"p-type, x=0.20, a-axis": {"input": [0.2, 1.0, 0.0, 1.0, 0.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [5.46e-06, 5.92e-06, 7.32e-06, 9.07e-06, 1.11e-05, 1.34e-05, 1.62e-05], "Seebeck_coefficient [V/K]": [0.000155, 0.000156, 0.00016299999999999998, 0.00016999999999999999, 0.000175, 0.000175, 0.00016999999999999999], "thermal_conductivity [W/m/K]": [1.5950600000000001, 1.56619, 1.4585700000000001, 1.37535, 1.30731, 1.2859200000000002, 1.29601]}, "p-type, x=0.20, c-axis": {"input": [0.2, 1.0, 0.0, 0.0, 1.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [6e-06, 6.52e-06, 8.04e-06, 9.97e-06, 1.22e-05, 1.4800000000000002e-05, 1.7899999999999998e-05], "Seebeck_coefficient [V/K]": [0.000155, 0.00016, 0.000165, 0.000175, 0.000181, 0.000182, 0.00017900000000000001], "thermal_conductivity [W/m/K]": [1.44125, 1.39125, 1.31724, 1.23914, 1.19038, 1.15791, 1.16078]}, "p-type, x=0.30, a-axis": {"input": [0.3, 1.0, 0.0, 1.0, 0.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [7.71e-06, 8.43e-06, 1.05e-05, 1.3000000000000001e-05, 1.55e-05, 1.8e-05, 2.0399999999999998e-05], "Seebeck_coefficient [V/K]": [0.00018700000000000002, 0.000188, 0.000197, 0.00020299999999999997, 0.00020099999999999998, 0.00019099999999999998, 0.000173], "thermal_conductivity [W/m/K]": [1.28147, 1.27502, 1.1997200000000001, 1.16088, 1.1395899999999999, 1.1732, 1.25506]}, "p-type, x=0.30, c-axis": {"input": [0.3, 1.0, 0.0, 0.0, 1.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [8.67e-06, 9.45e-06, 1.18e-05, 1.4499999999999998e-05, 1.75e-05, 2.0300000000000002e-05, 2.33e-05], "Seebeck_coefficient [V/K]": [0.00018999999999999998, 0.000194, 0.00020299999999999997, 0.00020899999999999998, 0.000206, 0.000199, 0.000184], "thermal_conductivity [W/m/K]": [1.20449, 1.1629200000000002, 1.1007799999999999, 1.05172, 1.04398, 1.0685200000000001, 1.12901]}, "p-type, x=0.40, a-axis": {"input": [0.4, 1.0, 0.0, 1.0, 0.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [1.22e-05, 1.34e-05, 1.67e-05, 2.0399999999999998e-05, 2.33e-05, 2.49e-05, 2.54e-05], "Seebeck_coefficient [V/K]": [0.000236, 0.00024300000000000002, 0.000246, 0.00023500000000000002, 0.000213, 0.00018899999999999999, 0.000157], "thermal_conductivity [W/m/K]": [1.0821100000000001, 1.0551, 1.00571, 1.0155100000000001, 1.0532700000000002, 1.16185, 1.35869]}, "p-type, x=0.40, c-axis": {"input": [0.4, 1.0, 0.0, 0.0, 1.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [1.32e-05, 1.4599999999999999e-05, 1.82e-05, 2.2100000000000002e-05, 2.54e-05, 2.7300000000000003e-05, 2.8100000000000002e-05], "Seebeck_coefficient [V/K]": [0.000223, 0.000229, 0.00023700000000000001, 0.00023500000000000002, 0.000221, 0.000197, 0.000167], "thermal_conductivity [W/m/K]": [0.9964799999999999, 0.9718100000000001, 0.93727, 0.93311, 0.9723799999999999, 1.0555, 1.21451]}, "p-type, x=0.46, a-axis": {"input": [0.46, 1.0, 0.0, 1.0, 0.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [1.6899999999999997e-05, 1.85e-05, 2.27e-05, 2.6700000000000002e-05, 2.8999999999999997e-05, 2.94e-05, 2.8800000000000002e-05], "Seebeck_coefficient [V/K]": [0.000245, 0.000251, 0.000256, 0.00023999999999999998, 0.000212, 0.000176, 0.00013700000000000002], "thermal_conductivity [W/m/K]": [0.9637, 0.9449799999999999, 0.92376, 0.9587100000000001, 1.0411, 1.19214, 1.36691]}, "p-type, x=0.46, c-axis": {"input": [0.46, 1.0, 0.0, 0.0, 1.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [1.87e-05, 2.05e-05, 2.54e-05, 2.98e-05, 3.27e-05, 3.35e-05, 3.3600000000000004e-05], "Seebeck_coefficient [V/K]": [0.00024300000000000002, 0.000248, 0.000252, 0.00024300000000000002, 0.000217, 0.000185, 0.000149], "thermal_conductivity [W/m/K]": [0.9949100000000001, 0.9786799999999999, 0.9462299999999999, 0.9574600000000001, 1.03985, 1.1547, 1.31073]}, "p-type, x=0.50, a-axis": {"input": [0.5, 1.0, 0.0, 1.0, 0.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [2.3899999999999998e-05, 2.6000000000000002e-05, 3.13e-05, 3.41e-05, 3.3799999999999995e-05, 3.17e-05, 2.9299999999999997e-05], "Seebeck_coefficient [V/K]": [0.000271, 0.000271, 0.000264, 0.00023199999999999997, 0.00018700000000000002, 0.000147, 0.00011399999999999999], "thermal_conductivity [W/m/K]": [0.9392299999999999, 0.95139, 0.97183, 1.0539, 1.2071399999999999, 1.38415, 1.5753]}, "p-type, x=0.50, c-axis": {"input": [0.5, 1.0, 0.0, 0.0, 1.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [2.5300000000000002e-05, 2.7699999999999996e-05, 3.3299999999999996e-05, 3.64e-05, 3.6200000000000006e-05, 3.4200000000000005e-05, 3.21e-05], "Seebeck_coefficient [V/K]": [0.000277, 0.000277, 0.000271, 0.000245, 0.000196, 0.00015900000000000002, 0.000125], "thermal_conductivity [W/m/K]": [0.8914200000000001, 0.88261, 0.89361, 0.96098, 1.09019, 1.25766, 1.4451399999999999]}, "n-type, x=0.30, a-axis": {"input": [0.3, 0.0, 1.0, 1.0, 0.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300, 25, 50, 100, 150, 200, 250, 300, 25, 50, 100, 150, 200, 250, 300, 25, 50, 100, 150, 200, 250, 300, 25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [9.88e-06, 1.0300000000000001e-05, 1.17e-05, 1.32e-05, 1.4599999999999999e-05, 1.59e-05, 1.67e-05, 9.43e-06, 9.89e-06, 1.12e-05, 1.27e-05, 1.4099999999999999e-05, 1.5399999999999998e-05, 1.63e-05, 1.06e-05, 1.12e-05, 1.25e-05, 1.4000000000000001e-05, 1.5399999999999998e-05, 1.66e-05, 1.74e-05, 7.8e-06, 8.15e-06, 9.3e-06, 1.06e-05, 1.2e-05, 1.3699999999999998e-05, 1.63e-05, 1.06e-05, 1.11e-05, 1.25e-05, 1.4099999999999999e-05, 1.5399999999999998e-05, 1.65e-05, 1.7100000000000002e-05], "Seebeck_coefficient [V/K]": [-0.00015, -0.000153, -0.000161, -0.000167, -0.000166, -0.00016, -0.000147, -0.000149, -0.000151, -0.000161, -0.000166, -0.000166, -0.00016, -0.000147, -0.000157, -0.000162, -0.000168, -0.000173, -0.00016999999999999999, -0.000162, -0.000148, -0.000128, -0.000129, -0.000138, -0.000144, -0.000148, -0.000148, -0.000144, -0.000165, -0.00016999999999999999, -0.000177, -0.00018, -0.000177, -0.000165, -0.000147], "thermal_conductivity [W/m/K]": [1.3072464959999999, 1.26958419, 1.229492057, 1.208838535, 1.222202579, 1.280518407, 1.38743076, 1.243998488, 1.221028643, 1.16058168, 1.140029713, 1.137611834, 1.17750683, 1.265759395, 1.197367549, 1.1779769820000001, 1.131924384, 1.114957637, 1.124652921, 1.17312934, 1.291896566, 1.272186425, 1.224343517, 1.15692851, 1.122133667, 1.146055122, 1.229780211, 1.342863449, 1.17154, 1.13883, 1.10369, 1.08189, 1.10733, 1.17275, 1.28179]}, "n-type, x=0.30, c-axis": {"input": [0.3, 0.0, 1.0, 0.0, 1.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300, 25, 50, 100, 150, 200, 250, 300, 25, 50, 100, 150, 200, 250, 300, 25, 50, 100, 150, 200, 250, 300, 25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [1.1400000000000001e-05, 1.1900000000000001e-05, 1.34e-05, 1.4999999999999999e-05, 1.66e-05, 1.7800000000000002e-05, 1.85e-05, 1.05e-05, 1.1e-05, 1.24e-05, 1.4099999999999999e-05, 1.56e-05, 1.6899999999999997e-05, 1.7800000000000002e-05, 1.2e-05, 1.25e-05, 1.39e-05, 1.55e-05, 1.7e-05, 1.8100000000000003e-05, 1.86e-05, 9.09e-06, 9.65e-06, 1.09e-05, 1.26e-05, 1.4099999999999999e-05, 1.6100000000000002e-05, 1.91e-05, 1.25e-05, 1.3000000000000001e-05, 1.47e-05, 1.65e-05, 1.8100000000000003e-05, 1.92e-05, 1.98e-05], "Seebeck_coefficient [V/K]": [-0.000153, -0.00015800000000000002, -0.000165, -0.000169, -0.000167, -0.000161, -0.000146, -0.00015, -0.000154, -0.000161, -0.000167, -0.000167, -0.000161, -0.000147, -0.000156, -0.000161, -0.000167, -0.000172, -0.00016999999999999999, -0.00016, -0.000146, -0.00011899999999999999, -0.000123, -0.000134, -0.000143, -0.000147, -0.000149, -0.000143, -0.000166, -0.000171, -0.000177, -0.00017900000000000001, -0.000175, -0.000162, -0.000145], "thermal_conductivity [W/m/K]": [1.197904317, 1.177250794, 1.131084096, 1.108000747, 1.125009531, 1.176035881, 1.274443842, 1.243998488, 1.204103493, 1.167835316, 1.135193956, 1.142447591, 1.175088951, 1.279057727, 1.181612713, 1.145255399, 1.116169547, 1.0955670690000001, 1.113745727, 1.17434125, 1.294320387, 1.094950197, 1.063417371, 1.0057884129999999, 0.9753429259999999, 0.994915025, 1.056893338, 1.1417057659999998, 1.16306, 1.14731, 1.10733, 1.08552, 1.1049, 1.16669, 1.28421]}, "n-type, x=0.40, a-axis": {"input": [0.4, 0.0, 1.0, 1.0, 0.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [1.0800000000000002e-05, 1.13e-05, 1.26e-05, 1.4000000000000001e-05, 1.55e-05, 1.6800000000000002e-05, 1.7800000000000002e-05], "Seebeck_coefficient [V/K]": [-0.000154, -0.00015900000000000002, -0.000166, -0.000171, -0.000174, -0.000167, -0.000154], "thermal_conductivity [W/m/K]": [1.158349847, 1.134292737, 1.086178517, 1.064527118, 1.063324262, 1.087381372, 1.177595535]}, "n-type, x=0.40, c-axis": {"input": [0.4, 0.0, 1.0, 0.0, 1.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [1.34e-05, 1.4000000000000001e-05, 1.55e-05, 1.73e-05, 1.89e-05, 2.0300000000000002e-05, 2.12e-05], "Seebeck_coefficient [V/K]": [-0.00015900000000000002, -0.000162, -0.000169, -0.000174, -0.000172, -0.000166, -0.000152], "thermal_conductivity [W/m/K]": [1.097004216, 1.06933854, 1.0320500190000002, 1.012804331, 1.015210042, 1.053701418, 1.146321292]}, "n-type, x=0.45, a-axis": {"input": [0.45, 0.0, 1.0, 1.0, 0.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [1.0199999999999999e-05, 1.07e-05, 1.2e-05, 1.3500000000000001e-05, 1.4800000000000002e-05, 1.62e-05, 1.77e-05], "Seebeck_coefficient [V/K]": [-0.00013700000000000002, -0.00014099999999999998, -0.000149, -0.000155, -0.000157, -0.00015800000000000002, -0.000151], "thermal_conductivity [W/m/K]": [1.217170195, 1.180563572, 1.131373424, 1.118789897, 1.15539652, 1.249200989, 1.375036254]}, "n-type, x=0.45, c-axis": {"input": [0.45, 0.0, 1.0, 0.0, 1.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [1.49e-05, 1.5399999999999998e-05, 1.7e-05, 1.88e-05, 2.02e-05, 2.1899999999999997e-05, 2.37e-05], "Seebeck_coefficient [V/K]": [-0.000138, -0.00014099999999999998, -0.000149, -0.000155, -0.000157, -0.000155, -0.000148], "thermal_conductivity [W/m/K]": [0.998674417, 0.971219451, 0.935756785, 0.927749086, 0.962067795, 1.042144781, 1.1267975959999998]}, "n-type, x=0.50, a-axis": {"input": [0.5, 0.0, 1.0, 1.0, 0.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [1.2e-05, 1.24e-05, 1.3699999999999998e-05, 1.53e-05, 1.6899999999999997e-05, 1.84e-05, 1.9600000000000002e-05], "Seebeck_coefficient [V/K]": [-0.000155, -0.00015900000000000002, -0.000167, -0.000173, -0.000175, -0.000171, -0.00015900000000000002], "thermal_conductivity [W/m/K]": [1.102852152, 1.08723442, 1.034374404, 1.0079443959999999, 1.0103471240000002, 1.03317304, 1.102852152]}, "n-type, x=0.50, c-axis": {"input": [0.5, 0.0, 1.0, 0.0, 1.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [1.3699999999999998e-05, 1.42e-05, 1.56e-05, 1.74e-05, 1.91e-05, 2.0600000000000003e-05, 2.1600000000000003e-05], "Seebeck_coefficient [V/K]": [-0.000157, -0.00016, -0.000168, -0.000174, -0.000175, -0.00016999999999999999, -0.00015800000000000002], "thermal_conductivity [W/m/K]": [1.089637148, 1.070415324, 1.02716622, 1.013951216, 1.001937576, 1.035575768, 1.10525488]}, "n-type, x=0.60, a-axis": {"input": [0.6, 0.0, 1.0, 1.0, 0.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [1.3800000000000002e-05, 1.4300000000000002e-05, 1.57e-05, 1.74e-05, 1.91e-05, 2.07e-05, 2.2100000000000002e-05], "Seebeck_coefficient [V/K]": [-0.000161, -0.000162, -0.000171, -0.00018, -0.00017900000000000001, -0.000176, -0.000165], "thermal_conductivity [W/m/K]": [1.131370613, 1.109752067, 1.0593087909999999, 1.038891274, 1.02567994, 1.037690244, 1.085731459]}, "n-type, x=0.60, c-axis": {"input": [0.6, 0.0, 1.0, 0.0, 1.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [1.4300000000000002e-05, 1.4800000000000002e-05, 1.63e-05, 1.8e-05, 1.98e-05, 2.14e-05, 2.27e-05], "Seebeck_coefficient [V/K]": [-0.00015800000000000002, -0.000162, -0.00016999999999999999, -0.000175, -0.000178, -0.000173, -0.000164], "thermal_conductivity [W/m/K]": [1.102545884, 1.083329398, 1.031685092, 1.0016593329999999, 0.9812418159999999, 0.9896490290000001, 1.035288183]}, "n-type, x=0.70, a-axis": {"input": [0.7, 0.0, 1.0, 1.0, 0.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [1.26e-05, 1.3000000000000001e-05, 1.4300000000000002e-05, 1.6e-05, 1.77e-05, 1.95e-05, 2.12e-05], "Seebeck_coefficient [V/K]": [-0.000153, -0.000156, -0.000165, -0.000172, -0.000176, -0.000177, -0.00016999999999999999], "thermal_conductivity [W/m/K]": [1.098903288, 1.08929539, 1.041255902, 1.000422338, 0.9764025940000001, 0.988412466, 1.02083912]}, "n-type, x=0.70, c-axis": {"input": [0.7, 0.0, 1.0, 0.0, 1.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [1.4499999999999998e-05, 1.4999999999999999e-05, 1.65e-05, 1.83e-05, 2.02e-05, 2.22e-05, 2.3800000000000003e-05], "Seebeck_coefficient [V/K]": [-0.000155, -0.00015900000000000002, -0.000167, -0.000174, -0.000178, -0.000176, -0.000169], "thermal_conductivity [W/m/K]": [1.056868736, 1.035250966, 0.9896134529999999, 0.957186798, 0.934368042, 0.942774952, 0.9836085170000001]}, "n-type, x=0.90, a-axis": {"input": [0.9, 0.0, 1.0, 1.0, 0.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [1.12e-05, 1.18e-05, 1.32e-05, 1.51e-05, 1.7100000000000002e-05, 1.93e-05, 2.1600000000000003e-05], "Seebeck_coefficient [V/K]": [-0.00014, -0.000143, -0.000153, -0.000162, -0.000168, -0.000172, -0.000172], "thermal_conductivity [W/m/K]": [1.18701705, 1.13627974, 1.061943683, 1.006486624, 0.9805280000000001, 0.9970471240000001, 1.06312362]}, "n-type, x=0.90, c-axis": {"input": [0.9, 0.0, 1.0, 0.0, 1.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [1.42e-05, 1.47e-05, 1.63e-05, 1.84e-05, 2.07e-05, 2.3100000000000002e-05, 2.58e-05], "Seebeck_coefficient [V/K]": [-0.00014099999999999998, -0.000143, -0.000154, -0.00016299999999999998, -0.000168, -0.000172, -0.000173], "thermal_conductivity [W/m/K]": [1.100881618, 1.065483495, 0.986427687, 0.936870315, 0.91209163, 0.930970628, 0.987607625]}, "n-type, x=0.99, a-axis": {"input": [0.99, 0.0, 1.0, 1.0, 0.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [1.4300000000000002e-05, 1.4800000000000002e-05, 1.62e-05, 1.8100000000000003e-05, 2.02e-05, 2.24e-05, 2.49e-05], "Seebeck_coefficient [V/K]": [-0.000146, -0.000151, -0.00015800000000000002, -0.000168, -0.000175, -0.000178, -0.000177], "thermal_conductivity [W/m/K]": [1.162320341, 1.122280937, 1.049267907, 0.9939193190000001, 0.9762548759999999, 0.997452208, 1.070465238]}, "n-type, x=0.99, c-axis": {"input": [0.99, 0.0, 1.0, 0.0, 1.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [1.62e-05, 1.6800000000000002e-05, 1.83e-05, 2.02e-05, 2.23e-05, 2.46e-05, 2.6899999999999997e-05], "Seebeck_coefficient [V/K]": [-0.000151, -0.000155, -0.00016299999999999998, -0.000171, -0.000178, -0.00017900000000000001, -0.000177], "thermal_conductivity [W/m/K]": [1.06575472, 1.0210047979999999, 0.956235174, 0.9067747340000001, 0.886755032, 0.902064216, 0.960945692]}, "n-type, x=1.05, a-axis": {"input": [1.05, 0.0, 1.0, 1.0, 0.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [1.34e-05, 1.39e-05, 1.5399999999999998e-05, 1.73e-05, 1.93e-05, 2.15e-05, 2.3800000000000003e-05], "Seebeck_coefficient [V/K]": [-0.00013700000000000002, -0.000139, -0.000148, -0.000157, -0.000164, -0.000168, -0.00016999999999999999], "thermal_conductivity [W/m/K]": [1.214177641, 1.171366285, 1.103581637, 1.046499828, 1.02509415, 1.036986193, 1.094068002]}, "n-type, x=1.05, c-axis": {"input": [1.05, 0.0, 1.0, 0.0, 1.0], "temperature [degC]": [25, 50, 100, 150, 200, 250, 300], "electrical_resistivity [Ohm m]": [1.77e-05, 1.84e-05, 2.02e-05, 2.25e-05, 2.49e-05, 2.7300000000000003e-05, 2.98e-05], "Seebeck_coefficient [V/K]": [-0.000136, -0.000139, -0.000147, -0.000157, -0.000164, -0.00016999999999999999, -0.000168], "thermal_conductivity [W/m/K]": [1.012012902, 0.9739583629999999, 0.916876554, 0.8764436059999999, 0.8574163359999999, 0.8716867890000001, 0.918065758]}};