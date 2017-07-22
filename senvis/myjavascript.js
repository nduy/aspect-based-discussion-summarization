//var len = undefined;
var dataset_options = {};
var nodesDataset = new vis.DataSet(dataset_options); 
var edgesDataset = new vis.DataSet(dataset_options);
var network = null;
var allNodes;
var allEdges;
var highlightActive = false;
var parsed_text;
var color_book; // save original color of all nodes
var seed = 2;
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

function setDefaultLocale() {
   var defaultLocal = navigator.language;
   var select = document.getElementById('locale');
   select.selectedIndex = 0; // set fallback value
   for (var i = 0, j = select.options.length; i < j; ++i) {
     if (select.options[i].getAttribute('value') === defaultLocal) {
       select.selectedIndex = i;
       break;
     }
   }
}
 
function destroy() {
   if (network !== null) {
     network.destroy();
     network = null;
   }
}
 
function draw() {
	destroy();
   // create a network
	var container = document.getElementById('mynetwork');
	//console.log(jsondata.nodes);
	//console.log(jsondata.edges);
	//Option for graph
	var options = {
	   autoResize: true,
		height: '100%',
	   width: '100%',
	   locale: document.getElementById('locale').value,
	   manipulation: {
       addNode: function (data, callback) {
         // filling in the popup DOM elements
         document.getElementById('node-operation').innerHTML = "Add Node";
         
         editNode(data, callback);
       },
       editNode: function (data, callback) {
         // filling in the popup DOM elements
         document.getElementById('node-operation').innerHTML = "Edit Node";
         editNode(data, callback);
       },
       addEdge: function (data, callback) {
         if (data.from == data.to) {
           var r = confirm("Do you want to connect the node to itself?");
           if (r != true) {
             callback(null);
             return;
           }
         }
         document.getElementById('edge-operation').innerHTML = "Add Edge";
         editEdgeWithoutDrag(data, callback);
       },
       editEdge: {
         editWithoutDrag: function(data, callback) {
           document.getElementById('edge-operation').innerHTML = "Edit Edge";
           editEdgeWithoutDrag(data,callback);
         }
       }
	   },
	   //locales: locales,
	   clickToUse: false,
	   physics: true,
	   
	   layout: {
	    improvedLayout:false,
	    randomSeed:seed
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
	//console.log(allNodes);
	allEdges = edgesDataset.get({returnType:"Object"});
	
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

function editNode(data, callback) {
   //console.log(data);
   document.getElementById('node-label').value = data.label;
   document.getElementById('node-id').value = data.id;
   document.getElementById('node-color').value = data.color ? color2hex(data.color.background) : 'white';
   document.getElementById('node-value').value = data.value ? data.value : 1;
   document.getElementById('node-saveButton').onclick = saveNodeData.bind(this, data, callback);
   document.getElementById('node-cancelButton').onclick = cancelNodeEdit.bind(this,callback);
   document.getElementById('node-popUp').style.display = 'block';
}

function clearNodePopUp() {
   document.getElementById('node-saveButton').onclick = null;
   document.getElementById('node-cancelButton').onclick = null;
   document.getElementById('node-popUp').style.display = 'none';
 }

function cancelNodeEdit(callback) {
   clearNodePopUp();
   callback(null);
   
 }

function saveNodeData(data, callback) {
	data.id = document.getElementById('node-label').value
			.concat('~x~x~')
			.concat(document.getElementById('node-id').value);
    data.label = document.getElementById('node-label').value;
    var clr = document.getElementById('node-color').value;
    if (data.color){
		data.color.background = clr;
	} else {
		data['color'] = {'background':clr, 'border' : clr};
	}
	data.color.border = clr;
	data.value = document.getElementById('node-value').value;
	var new_title = "[edited] <br>*Value: ".concat(data.value)
											.concat(' <br>*Score: UNKNOWN')
											.concat(' <br>*Color: ').concat(data.color.background)
											.concat(data.title? '<br>[original] <br>'.concat(data.title) : '<br>[New node]')
											;
	data.title = new_title;
	if (allNodes[data.id]){
		allNodes[data.id]['value'] = data.value;
		allNodes[data.id]['title'] = new_title;
	} else {
		allNodes[data.id]=
			{
				"color": clr,
				"id": data.id,
				"label": data.label,
				"title": new_title,
				"value": data.value
			};
		}
	allNodes[data.id]['color'] = clr;
	// update color book
   color_book[data.id] = clr;
   clearNodePopUp();
   //console.log(data);
   callback(data);
}

function editEdgeWithoutDrag(data, callback) {
   // filling in the popup DOM elements
   document.getElementById('edge-label').value = allEdges[data.id] ? allEdges[data.id]['label'] : 'new label';
   document.getElementById('edge-value').value = allEdges[data.id]? allEdges[data.id]['value'] : 1;
   document.getElementById('edge-saveButton').onclick = saveEdgeData.bind(this, data, callback);
   document.getElementById('edge-cancelButton').onclick = cancelEdgeEdit.bind(this,callback);
   document.getElementById('edge-popUp').style.display = 'block';
}

function clearEdgePopUp() {
   document.getElementById('edge-saveButton').onclick = null;
   document.getElementById('edge-cancelButton').onclick = null;
   document.getElementById('edge-popUp').style.display = 'none';
}

function cancelEdgeEdit(callback) {
   clearEdgePopUp();
   callback(null);
   
}

function saveEdgeData(data, callback) {
   if (typeof data.to === 'object')
     data.to = data.to.id
   if (typeof data.from === 'object')
     data.from = data.from.id
   data.label = document.getElementById('edge-label').value;
   data.value = document.getElementById('edge-value').value;
   var new_title = "[edited] <br>*Value: ".concat(data.value)
					.concat(' <br>*Score: UNKNOWN')
					.concat(data.title ? '<br>[original] <br>'.concat(data.title) : '<br>[New edge]');
   data.title = new_title;
   if (allEdges[data.id]) {
	   allEdges[data.id]['value'] = data.value;
	   allEdges[data.id]['label'] = data.label;
	   allEdges[data.id]['title'] = new_title;
   } else {
	   allEdges[data.id] = {'id': data.id,
							'label': data.label,
							'title': new_title
		                    };
   }
   clearEdgePopUp();
   callback(data);
}

function init() {
   setDefaultLocale();
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

function drawFromJS() {
   jtextArea = document.getElementById("jsonarea");
	//jtextArea.value="Input the json";
	// Event of pressing input json
	document.getElementById("submitbutton").addEventListener("click", function(){
		// Hide the language selection combobox
		document.getElementById("locale").disabled = true;
		// Get the data
		jtext = jtextArea.value;
		if (jtext== "") {
			alert("Please enter json document! oyo")
		}
		else {
			try {
				parsed_text = JSON.parse(jtext);
				nodesDataset.clear();
   			edgesDataset.clear();
				nodesDataset.add(parsed_text.nodes);
				edgesDataset.add(parsed_text.edges);
				draw();
			}
			catch(err) {
				console.log(err.message);
				alert("Unable to draw the network!", console.log(err.message));
			};
		};
		
	});
}

//Function to convert hex format to a rgb color
function color2hex(rgb){
 ori = rgb;
 rgb = rgb.match(/^rgba?[\s+]?\([\s+]?(\d+)[\s+]?,[\s+]?(\d+)[\s+]?,[\s+]?(\d+)[\s+]?/i);
 return (rgb && rgb.length === 4) ? "#" +
  ("0" + parseInt(rgb[1],10).toString(16)).slice(-2) +
  ("0" + parseInt(rgb[2],10).toString(16)).slice(-2) +
  ("0" + parseInt(rgb[3],10).toString(16)).slice(-2) : ori;
}



$(document).ready(function () {
	drawFromJS();
});
