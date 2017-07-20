//var len = undefined;
var dataset_options = {};
var nodesDataset = new vis.DataSet(dataset_options); 
var edgesDataset = new vis.DataSet(dataset_options);
var network;
var allNodes;
var highlightActive = false;
var parsed_text;
var color_book; // save original color of all nodes
var locales = {
  en: {
    edit: 'Edit',
    del: 'Delete selected',
    back: 'Back',
    addNode: 'Add Node',
    addEdge: 'Add Edge',
    editNode: 'Edit Node',
    editEdge: 'Edit Edge',
    addDescription: 'Click in an empty space to place a new node.',
    edgeDescription: 'Click on a node and drag the edge to another node to connect them.',
    editEdgeDescription: 'Click on the control points and drag them to a node to connect to it.',
    createEdgeError: 'Cannot link edges to a cluster.',
    deleteClusterError: 'Clusters cannot be deleted.',
    editClusterError: 'Clusters cannot be edited.'
  }
}

function draw() {
   // create a network
	var container = document.getElementById('mynetwork');
	//console.log(jsondata.nodes);
	//console.log(jsondata.edges);
	//Option for graph
	var options = {
	   autoResize: true,
		height: '100%',
	   width: '100%',
	   locale: 'en',
	   locales: locales,
	   clickToUse: false,
	   physics: true,
	   
	   layout: {
	    improvedLayout:false
	  	},
	   
	   interaction:{
			hover:true,
			tooltipDelay: 200
		},
	   nodes: {
			shape: 'dot',
			font: {
	                size: 28,
	                color: '#ffffff'
	            },
	     scaling: {
			  min: 10,
			  max: 30,
			  label: {
				min: 10,
				max: 30,
				drawThreshold: 12,
				maxVisible: 20
			  }
		  },  
	    },
	    edges: {
	     // smooth: {
	     //   type: 'continuous'
	     // }
			color: {inherit: 'both'},
			smooth: false
	    },
	    configure:function (option, path) {
	      if (path.indexOf('smooth') !== -1 || option === 'smooth') {
	        return true;
	      }
	      return false;
	    }
		  
	};	
	
	var data = {
	  nodes: nodesDataset,
	  edges: edgesDataset
		}; 
	
	network = new vis.Network(container, data, options);
	// get a JSON object
	allNodes = nodesDataset.get({returnType:"Object"});
	
	// Save all original color to color box
	color_book = {};
	for (var nodeId in allNodes) {
     color_book[nodeId] = allNodes[nodeId].color;
   }
	
/*	network.on("click", function (params) {
        params.event = "[original event]";
        //document.getElementById('eventSpan').innerHTML = '<h2>Click event:</h2>' + JSON.stringify(params, null, 4);
        console.log('click event, getNodeAt returns: ' + this.getNodeAt(params.pointer.DOM));
        neighbourhoodHighlight(params);
        
    });

	network.on("click",neighbourhoodHighlight);

	network.on("doubleClick", function (params) {
        params.event = "[original event]";
        //document.getElementById('eventSpan').innerHTML = '<h2>doubleClick event:</h2>' + JSON.stringify(params, null, 4);
    });
	network.on("oncontext", function (params) {
        params.event = "[original event]";
        //document.getElementById('eventSpan').innerHTML = '<h2>oncontext (right click) event:</h2>' + JSON.stringify(params, null, 4);
    });
	network.on("dragStart", function (params) {
        params.event = "[original event]";
        //document.getElementById('eventSpan').innerHTML = '<h2>dragStart event:</h2>' + JSON.stringify(params, null, 4);
    });
	network.on("dragging", function (params) {
        params.event = "[original event]";
        //document.getElementById('eventSpan').innerHTML = '<h2>dragging event:</h2>' + JSON.stringify(params, null, 4);
    });
	network.on("dragEnd", function (params) {
        params.event = "[original event]";
        //document.getElementById('eventSpan').innerHTML = '<h2>dragEnd event:</h2>' + JSON.stringify(params, null, 4);
    });
	network.on("zoom", function (params) {
        //document.getElementById('eventSpan').innerHTML = '<h2>zoom event:</h2>' + JSON.stringify(params, null, 4);
    });
	network.on("showPopup", function (params) {
        //document.getElementById('eventSpan').innerHTML = '<h2>showPopup event: </h2>' + JSON.stringify(params, null, 4);
    });
	network.on("hidePopup", function () {
        console.log('hidePopup Event');
    });
	network.on("select", function (params) {
        console.log('select Event:', params);
    });
	network.on("selectNode", function (params) {
        console.log('selectNode Event:', params);
    });
	network.on("selectEdge", function (params) {
        console.log('selectEdge Event:', params);
    });
	network.on("deselectNode", function (params) {
        console.log('deselectNode Event:', params);
    });
	network.on("deselectEdge", function (params) {
        console.log('deselectEdge Event:', params);
    });
	network.on("hoverNode", function (params) {
        console.log('hoverNode Event:', params);
    });
	network.on("hoverEdge", function (params) {
        console.log('hoverEdge Event:', params);
    });
	network.on("blurNode", function (params) {
        console.log('blurNode Event:', params);
    });
	network.on("blurEdge", function (params) {
        console.log('blurEdge Event:', params);
    });
*/
	network.on("hoverNode", function (params) {
        neighbourhoodHighlight(params);
    });
}
 
function neighbourhoodHighlight(params) {
//console.log(params);
 // if something is selected:
 if (params.node) {
   highlightActive = true;
   var i,j;
   var selectedNode = params.node;
   var degrees = 2;

   // mark all nodes as hard to read.
   for (var nodeId in allNodes) {
     allNodes[nodeId].color = 'rgba(200,200,200,0.5)';
     if (allNodes[nodeId].hiddenLabel === undefined) {
       allNodes[nodeId].hiddenLabel = allNodes[nodeId].label;
       allNodes[nodeId].label = undefined;
     }
   }
   var connectedNodes = network.getConnectedNodes(selectedNode);
   var allConnectedNodes = [];

   // get the second degree nodes
   for (i = 1; i < degrees; i++) {
     for (j = 0; j < connectedNodes.length; j++) {
       allConnectedNodes = allConnectedNodes.concat(network.getConnectedNodes(connectedNodes[j]));
     }
   }

   // all second degree nodes get a different color and their label back
   for (i = 0; i < allConnectedNodes.length; i++) {
     allNodes[allConnectedNodes[i]].color = 'rgba(150,150,150,0.75)';
     if (allNodes[allConnectedNodes[i]].hiddenLabel !== undefined) {
       allNodes[allConnectedNodes[i]].label = allNodes[allConnectedNodes[i]].hiddenLabel;
       allNodes[allConnectedNodes[i]].hiddenLabel = undefined;
     }
   }

   // all first degree nodes get their own color and their label back
   for (i = 0; i < connectedNodes.length; i++) {
     allNodes[connectedNodes[i]].color = color_book[connectedNodes[i]];
     if (allNodes[connectedNodes[i]].hiddenLabel !== undefined) {
       allNodes[connectedNodes[i]].label = allNodes[connectedNodes[i]].hiddenLabel;
       allNodes[connectedNodes[i]].hiddenLabel = undefined;
     }
   }

   // the main node gets its own color and its label back.
   allNodes[selectedNode].color = color_book[selectedNode];
   if (allNodes[selectedNode].hiddenLabel !== undefined) {
     allNodes[selectedNode].label = allNodes[selectedNode].hiddenLabel;
     allNodes[selectedNode].hiddenLabel = undefined;
   }
 }
 else if (highlightActive === true) {
   // reset all nodes
   for (var nodeId in allNodes) {
     allNodes[nodeId].color = undefined;
     if (allNodes[nodeId].hiddenLabel !== undefined) {
       allNodes[nodeId].label = allNodes[nodeId].hiddenLabel;
       allNodes[nodeId].hiddenLabel = undefined;
     }
   }
   highlightActive = false
 }

 // transform the object into an array
 var updateArray = [];
 for (nodeId in allNodes) {
   if (allNodes.hasOwnProperty(nodeId)) {
     updateArray.push(allNodes[nodeId]);
   }
 }
 nodesDataset.update(updateArray);
}


$(document).ready(function () {
	jtextArea = document.getElementById("jsonarea");
	//jtextArea.value="Input the json";
	// Event of pressing input json
	document.getElementById("submitbutton").addEventListener("click", function(){
		jtext = jtextArea.value;
		if (jtext== "") {
			alert("Please enter json document! oyo")
		}
		else {
			try {
				parsed_text = JSON.parse(jtext);
				nodesDataset.add(parsed_text.nodes);
				edgesDataset.add(parsed_text.edges);
				draw();
			}
			catch(err) {
				console.log(err.message);
				alert(console.log(err.message));
			};
		};
		
	});
});