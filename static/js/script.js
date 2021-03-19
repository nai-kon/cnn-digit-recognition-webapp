

let cvsIn = document.getElementById("inputimg");
let ctxIn = cvsIn.getContext("2d");
let divOut = document.getElementById("pred");
let svgGraph = null;
let mouselbtn = false;


// initilize
window.onload = ()=>{
    
    ctxIn.fillStyle = "white";
    ctxIn.fillRect(0, 0, cvsIn.width, cvsIn.height);
    ctxIn.lineWidth = 7;
    ctxIn.lineCap = "round";
    
    initProbGraph();
}

function initProbGraph(){

    const dummyData = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]; // dummy data for initialize graph
    const margin = { top: 10, right: 10, bottom: 10, left: 20 }
        , width = 250, height = 196;

    let yScale = d3.scaleLinear()
        .domain([9, 0])
        .range([height, 0]);
    
    svgGraph = d3.select("#probGraph")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    svgGraph.append("g")
        .attr("class", "y axis")
        .call(d3.axisLeft(yScale));

    const barHeight = 20
    svgGraph.selectAll("svg")
        .data(dummyData)
        .enter()
        .append("rect")
        .attr("y", (d,i)=>(yScale(i) - barHeight / 2))
        .attr("height", barHeight)
        .style("fill", "green")
        .attr("x", 0)
        .attr("width", (d)=>d * 2)
        .call(d3.axisLeft(yScale));
}

cvsIn.addEventListener("mousedown", (e)=>{ 

    if(e.button == 0){
        let rect = e.target.getBoundingClientRect();
        let x = e.clientX - rect.left;
        let y = e.clientY - rect.top;
        mouselbtn = true;
        ctxIn.beginPath();
        ctxIn.moveTo(x, y);
    }   
    else if(e.button == 2){
        onClear();  // clear by mouse right button
    }
});

cvsIn.addEventListener("mouseup", (e)=>{ 
    if(e.button == 0){
        mouselbtn = false; 
        onRecognition();
    }
});

cvsIn.addEventListener("mousemove", (e)=>{
    let rect = e.target.getBoundingClientRect();
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;
    if(mouselbtn){
        ctxIn.lineTo(x, y);
        ctxIn.stroke();
    }
});

cvsIn.addEventListener("touchstart", (e)=>{
    // for touch device
    if (e.targetTouches.length == 1) {
        let rect = e.target.getBoundingClientRect();
        let touch = e.targetTouches[0];
        let x = touch.clientX - rect.left;
        let y = touch.clientY - rect.top;
        ctxIn.beginPath();
        ctxIn.moveTo(x, y);
    }
});

cvsIn.addEventListener("touchmove", (e)=>{
    // for touch device
    if (e.targetTouches.length == 1) {
        let rect = e.target.getBoundingClientRect();
        let touch = e.targetTouches[0];
        let x = touch.clientX - rect.left;
        let y = touch.clientY - rect.top;
        ctxIn.lineTo(x, y);
        ctxIn.stroke();
        e.preventDefault();
    }
});

cvsIn.addEventListener("touchend", (e)=>onRecognition());

cvsIn.addEventListener("contextmenu", (e)=>e.preventDefault());

document.getElementById("clearbtn").onclick = onClear;
function onClear(){
    mouselbtn = false;
    ctxIn.fillStyle = "white";
    ctxIn.fillRect(0, 0, cvsIn.width, cvsIn.height);
    ctxIn.fillStyle = "black";
}

// post digit to server for recognition
function onRecognition() {
    console.time("time");

    cvsIn.toBlob((blob)=>{
        let form = new FormData();
        form.append('img', blob, "dummy.png")

        $.ajax({
            url: "./DigitRecognition",
            type: "POST",
            data: form,
            processData: false,
            contentType: false,
        })
        .then(
            (data)=>showResult(JSON.parse(data)),
            ()=>alert("error")
        )    
    })

    console.timeEnd("time");
}


function showResult(res){

    divOut.textContent = res.pred;

    document.getElementById("prob").innerHTML = 
        "Probability : " + res.probs[res.pred].toFixed(2) + "%";    

    svgGraph.selectAll("rect")
        .data(res.probs)
        .transition()
        .duration(300)
        .style("fill", (d, i) => i == res.pred ? "blue":"green")
        .attr("width", (d) => d * 2)
}
