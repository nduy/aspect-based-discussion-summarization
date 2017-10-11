//var len = undefined;
var dataset_options = {};
var nodesDataset = new vis.DataSet(dataset_options); 
var edgesDataset = new vis.DataSet(dataset_options);
var commentsDict = {}; // Store all comment
var network = null;
var data = null;
var jtext = null; // Raw graph description content
var allNodes;
var allEdges;
var highlightActive = false;
var parsed_text;
var color_book; // save original color of all nodes
var seed = 2;
var clustered = false;
var cluster_ids = null;
var enableLoading = false;
var enableSmily = false;
var groupComments = null; // save nodeid of each comments i
var group_members = null;
var n_comments = 20;
var cluster_radius = 400; // radius of each cluster
var selected_cluster = null; // selected cluster when right click, store so that can triggle command when context menu is sellected
var color_scale = scale = chroma.scale(['red', 'yellow']);

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
	enableLoading = true;
	enableSmily = true;
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
		   },
		   deleteNode: function (data, callback) {
			 // Remove the node from all nodes and color box
			 //console.log(allNodes[data.nodes[0]]);
			 delete allNodes[data.nodes[0]]; 
			 //console.log(allNodes[data.nodes[0]]);
			 //console.log(color_book[data.nodes[0]]);
			 delete color_book[data.nodes[0]];
			 //console.log(color_book[data.nodes[0]]);
			 //console.log(data);
			 callback(data);
		   }
	   },
	   //locales: locales,
	   clickToUse: false,
	   physics: {
		    enabled: true,
			barnesHut: {
			  centralGravity: 0.01,
			  springLength: 1,
			  springConstant: 0.01,
			  damping: 0.22,
			  avoidOverlap: 0.4,
			},
			maxVelocity: 10,
			minVelocity: 3,
			adaptiveTimestep: true,
			stabilization:{
				enabled: true,
				iterations: 1000,
			}
		  },
	   layout: {
	    improvedLayout:false,
	    randomSeed:seed
	  	},
	   
	   interaction:{
			hover:true,
			tooltipDelay: 200,
			navigationButtons: true,
			keyboard: true,
		},
	   nodes: {
		shape: 'square',
		font: {
				size: 28,
				color: '#ffffff'
			},
	     scaling: {
			  min: 20,
			  max: 40,
			  label: {
				min: 20,
				max: 40,
				drawThreshold: 12,
				maxVisible: 30
			  }
		  },  
	    },
	    edges: {
			arrows: {
			  to:     {enabled: true, scaleFactor:1, type:'arrow'},
			  middle: {enabled: false, scaleFactor:1, type:'arrow'},
			  from:   {enabled: false, scaleFactor:1, type:'arrow'}
			},
			color: {inherit: 'both'},
			smooth: {
			  enabled: true,
			  type: "dynamic",
			  roundness: 0.2
			},
			scaling:{
			  min: 1,
			  max: 10,
			  label: {
				enabled: true,
				min: 4,
				max: 20,
				maxVisible: 10,
				drawThreshold: 1
			  },
			  customScalingFunction: function (min,max,total,value) {
				if (max === min) {
				  return 0.5;
				}
				else {
				  var scale = 1 / (max - min);
				  return Math.max(0,(value - min)*scale);
				}
			  }
			},
	    },
	    configure:function (option, path) {
	      if (path.indexOf('smooth') !== -1 || option === 'smooth') {
	        return true;
	      }
	      return false;
	    },
	    groups: {
            central: {
                color: {background:'red',border:'white'},
                shape: 'star'
            },
            article: {
                color: {background:'red',border:'white'},
                shape: 'diamond'
            },
            comment: {
            color: {background:'red',border:'white'},
                shape: 'dot'
            }
        }
		  
	};	
	
	// Re arrange the comment to be a line
    y_pos = n_comments/2*-30;	// Starting pos
    x_pos = $('#mynetwork').width()*0.8;
    // alert(x_pos);
    for (let node_id of nodesDataset.getIds()) {
     if (node_id.match(/^comment~[\d]+$/i)) {
		// console.log(node_id);
		nodesDataset.update({id: node_id, y: y_pos, x : x_pos, size: 400, margin: {left: 100, right: 100}});
		y_pos+= 30;
     }
   }
	// Group to data
	data = {
	  nodes: nodesDataset,
	  edges: edgesDataset
		}; 
	// draw the network
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
   
	// Save all ClusterID -> commetn
	groupComments = {};
	for (var edgeId in allEdges) {
		if (edgeId.match(/^cmn\d+$/i)){ // is a cluster-comment edge
			// console.log(allEdges[edgeId]);
			if (!groupComments[allEdges[edgeId].from]){
				groupComments[allEdges[edgeId].from] = [allEdges[edgeId].to];
			} else {
				groupComments[allEdges[edgeId].from].push(allEdges[edgeId].to);
			}
		}
	}
	// console.log(groupComments);
	network.on("startStabilizing", function (params) {
		// document.getElementById('eventSpan').innerHTML = '<h3>Starting Stabilization</h3>';
		//console.log("started")
		if (enableLoading){
			$("#comments-box").html("<div class=\"welcome w3-spin\"><img src=\"img/loading.png\" alt=\"drawing...\" style=\"position: absolute; top: 0; bottom:0; left: 0; right:0; margin: auto;\"></div>");
			enableLoading = false;
		}
		
	});
/*  network.on("stabilizationProgress", function (params) {
    document.getElementById('eventSpan').innerHTML = '<h3>Stabilization progress</h3>' + JSON.stringify(params, null, 4);
    console.log("progress:",params);
  });
  network.on("stabilizationIterationsDone", function (params) {
    document.getElementById('eventSpan').innerHTML = '<h3>Stabilization iterations complete</h3>';
    console.log("finished stabilization interations");
  });
 */
	network.on("stabilized", function (params) {
	// document.getElementById('eventSpan').innerHTML = '<h3>Stabilized!</h3>' + JSON.stringify(params, null, 4);
	//console.log("stabilized!", params);
	if (enableSmily){
		// $("#comments-box").html("<div class=\"welcome w3-spin\">:)</div>");
		showAllComments();
		enableSmily = false;
	}
	});
  
	network.on("selectNode", function(params) {
	// console.log(params);
	if (params.nodes.length == 1) {
	 // console.log(network.isCluster(params.nodes[0]));
	 if (network.isCluster(params.nodes[0]) == true) {
		  
		  // $('.context-menu-one').contextMenu();
		  //$('.context-menu-one').trigger("contextmenu");
		  /*
		  // network.physics.options.enabled = false;
		  network.openCluster(params.nodes[0]);
		  // Remove node from dataset
		  nodesDataset.remove(params.nodes[0]);
		  delete allNodes[params.nodes[0]];
		  cluster_ids.delete(params.nodes[0]);
		  delete group_members[params.nodes[0]];
		  // console.log(cluster_ids);
		  if (cluster_ids.size == 0){
			clustered = false;
			$("#clusterButton").prop('value', 'Cluster nodes by topics'); 
		  }
		  // Now re-arange stuff at the center
		  
		  var central_x = allNodes[params.nodes[0]].x;
		  var central_y = allNodes[params.nodes[0]].y;
		  console.log(group_members[params.nodes[0]]);
		  var delta_phi = 360/group_members[params.nodes[0]].ids.length;
		  var phi = 0; //initialize phi at zero
		  for (let nodeID of group_members[params.nodes[0]].ids){
				position = rotate(0, 0, 0, 0 + cluster_radius, phi);
				phi += delta_phi;
				allNodes[nodeID].x = position[0];
				allNodes[nodeID].y = position[1];
				nodesDataset.update({ id: nodeID, x:  position[0], y:  position[1]});
		  }
		  network.physics.options.enabled = true;
		  */
		  console.log('');
	  }
	}
	});
	
	network.on("oncontext", function (params) {
        selected_nodeid = network.getNodeAt({x: params.event.layerX, y: params.event.layerY});
        if (selected_nodeid){
			if (network.isCluster(selected_nodeid)) {
				params.event.preventDefault();
				selected_cluster = selected_nodeid;
				$('.context-menu-one').contextMenu({x: params.event.layerX-200, y: params.event.layerY});
			}
		}
        //document.getElementById('eventSpan').innerHTML = '<h2>oncontext (right click) event:</h2>' + JSON.stringify(params, null, 4);
    });
	
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
        // Hightlight the neighbour nodes
        neighbourhoodHighlight(params);
        // If the nodes is comment node, then display the comment
        node_id = params.node;
        // console.log(node_id);
        // console.log(node_id.match(/^\[[α-ωΑ-Ω]\]\s/i));
        if (node_id.match(/^comment~[\d]+$/i)){ // Check if it is a comment
			// Clear the comment show area
			$("#comments-box").html("<div class=\"block-item\">" + commentsDict[node_id] + "</div>");
		} else if (network.isCluster(node_id)){ // is a Cluster node -> display text in clusters
			// Find all nodes id of the nodes that it connected with
			html_content = "";
			//console.log(groupComments[node_id]);
			// for (let comment_id of groupComments[node_id]){
			//	if (commentsDict[comment_id]){
			//		html_content = html_content + "<div class=\"block-item\">" + commentsDict[comment_id] + "</div>"; ///////////////////////////////////////////////////////////
			//	}	
			//}
			for (var comment_id in commentsDict){
				// console.log($.inArray(comment_id, group_members[node_id].ids));
				//console.log(group_members[node_id].ids);
				var condition = edgesDataset.get({
					filter: function (item) {
						return item.to == comment_id && ($.inArray(item.from, group_members[node_id].ids)>-1);
				  }
				});
				//console.log(condition);
				if (condition.length>0){
					html_content = html_content + "<div class=\"block-item\">" + commentsDict[comment_id] + "</div>";
				}
			}
			$("#comments-box").html(html_content); // Update to page
		}
		
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
	data.id = document.getElementById('node-id').value;
    data.label = document.getElementById('node-label').value;
    var clr = color_book[data.id] ? color_book[data.id] : 'white'
    //document.getElementById('node-color').value;
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
		data.id = document.getElementById('node-label').value
			.concat('~x~x~').concat(data.id );
		allNodes[data.id]=
			{
				"color": clr,
				"id": data.id,
				"label": data.label,
				"title": new_title,
				"value": data.value,
				"group": data.group
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
   document.getElementById('edge-label').value = allEdges[data.id] ? allEdges[data.id]['label'] : '';
   document.getElementById('edge-value').value = allEdges[data.id] ? allEdges[data.id]['value'] : 1;
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
 network.physics.options.enabled = false;
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
       // allNodes[nodeId].label = undefined; /////////////////////  [W]
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
	 if (allNodes[allConnectedNodes[i]] !== undefined) {
		if (allNodes[allConnectedNodes[i]].color !== undefined) {
			allNodes[allConnectedNodes[i]].color = 'rgba(150,150,150,0.75)';
		}
		if (allNodes[allConnectedNodes[i]].hiddenLabel !== undefined) {
			allNodes[allConnectedNodes[i]].label = allNodes[allConnectedNodes[i]].hiddenLabel;
			allNodes[allConnectedNodes[i]].hiddenLabel = undefined;
		}
	}
		   
		 
   }

   // all first degree nodes get their own color and their label back
   for (i = 0; i < connectedNodes.length; i++) {
     if (allNodes[connectedNodes[i]] !== undefined){
		 allNodes[connectedNodes[i]].color = color_book[connectedNodes[i]];
		 if (allNodes[connectedNodes[i]].hiddenLabel !== undefined) {
		   allNodes[connectedNodes[i]].label = allNodes[connectedNodes[i]].hiddenLabel;
		   allNodes[connectedNodes[i]].hiddenLabel = undefined;
		 }
	 }
   }

   // the main node gets its own color and its label back.
   if (allNodes[selectedNode] !== undefined){
		allNodes[selectedNode].color = color_book[selectedNode];
	    if (allNodes[selectedNode].hiddenLabel !== undefined) {
		 allNodes[selectedNode].label = allNodes[selectedNode].hiddenLabel;
		 allNodes[selectedNode].hiddenLabel = undefined;
	    }
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
     // updateArray.push(allNodes[nodeId]);
     updateArray.push({id: allNodes[nodeId].id, color: allNodes[nodeId].color});
     // nodesDataset.update({id: allNodes[nodeId].id, color: allNodes[nodeId].color});
   }
 }
 //console.log(updateArray);
 nodesDataset.update(updateArray);
 network.physics.options.enabled = true;
}

function drawFromJS() {
   //jtextArea = document.getElementById("jsonarea");
	//jtextArea.value="Input the json";
	// Event of pressing input json
	document.getElementById("submitbutton").addEventListener("click", function(){
		// Hide the language selection combobox
		document.getElementById("locale").disabled = true;
		// Get the data
		// jtext = jtextArea.value;
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
				n_comments = parsed_text.summary.n_comments;
				// Read the comments
				for (let item of parsed_text.comments){
					commentsDict[item.id] = item.label;
				}
				// console.log(commentsDict);
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

// Handle cluster/expand button
function handleExpandCluster(){
	if (clustered == false){
		enableLoading = true;
		enableSmily = true;
		network.physics.options.stabilization.enabled  = false;
		clusterByCid();
		clustered = true;
		$("#clusterButton").prop('value', 'Expand all clusters'); 
	} else {
		clustered = false;
		// console.log(cluster_ids);
		for (let nodeId of cluster_ids) {
		  //if (network.isCluster(nodeId) == true) {
		  // network.physics.options.enabled = false;
		  //console.log(network.physics.options);
		  network.openCluster(nodeId);
		  nodesDataset.remove(nodeId);
		  //console.log(network.physics.options);
		  //network.physics.options.enabled = true;
		  //}

		}
		// Reset all records
		cluster_ids = null;
		group_members = null;
		$("#clusterButton").prop('value', 'Cluster nodes by topics'); 
		console.log('0000-->');
		network.physics.options.stabilization.enabled  = true;
		
	}
}

// Cluster the group
function clusterByCid() {
  network.setData(data);
  cluster_ids = new Set();
  group_members = {}
  // Arrange nodes to clusters
  for (key in allNodes){
	  item = allNodes[key];
	  // console.log(item);
	  if (item.cid != undefined){
	  	  cluster_ids.add(item.cid);
	  	  if (!(item.cid in group_members)){  // the cluster id wasn't there before
			  group_members[item.cid] = {count: item.value, members: [item.label], ids: [key]};
		  } else {
			  group_members[item.cid].count = group_members[item.cid].count + item.value;
			  group_members[item.cid].members.push(item.label); // Add member to group
			  group_members[item.cid].ids.push(key); // Add id to group
		  }
			
	  }
  }
  // Coumpute weighted-average sentiment score
  for (var cid in group_members){
	 var avg = 0;
	 var total_count = group_members[cid].count;
	 for (let nodeid of group_members[cid].ids){
		// console.log(allNodes[nodeid].title.match(/Sen_Score: (-?[\d.]+)/i)[1]);
		avg += allNodes[nodeid].value/total_count*parseFloat(allNodes[nodeid].title.match(/Sen_Score: (-?[\d.]+)/i)[1]);
	 }
	 avg = Math.round(avg*10000)/10000;
	 group_members[cid].avg_sen_score = avg;// Do some sentiment composition here
	 group_members[cid].color = color_scale((avg+1)/2).hex();
	 var color = color_scale((avg+1)/2).hex();
	 color_book[cid] = color;
	 // allNodes[cid].color = color;
	 nodesDataset.update({id: cid, color: color});
  }
  
  // console.log(group_members);
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
  var clusterOptionsByData;
  var y_pos = -250*cluster_ids.size/2;
  for (let cid of cluster_ids.values()){
	  y_pos = y_pos + 250;
	  // console.log(y_pos);
	  clusterOptionsByData = {
		  joinCondition: function (childOptions) {
			  return childOptions.cid == cid; // the color is fully defined in the node.
		  },
		  /*processProperties: function (clusterOptions, childNodes, childEdges) {
			  
			  var totalMass = 0;
			  for (var i = 0; i < childNodes.length; i++) {
				  totalMass += childNodes[i].mass;
			  }
			  clusterOptions.mass = totalMass;
			  return clusterOptions;
			 
		  }, */
		  clusterNodeProperties: {id: cid, 
			  borderWidth: 1, shape: 'box', 
			  label: cid,
			  title: '☉ Nodes: <br> - ' + group_members[cid].members.join('<br> - ') + '<br>Σ Total count: '+ group_members[cid].count + '<br>' + ((group_members[cid].avg_sen_score<0) ? '☹': '☺') + ' W.mean sen.: ' + group_members[cid].avg_sen_score,
			  x : $('#mynetwork').width()*-0.8,
			  y : y_pos,
			  fixed: { y: true, y: true},
			  value: group_members[cid].count,
			  }
	  }; 
	  nodesDataset.update(clusterOptionsByData['clusterNodeProperties']);
	  network.cluster(clusterOptionsByData);
	  allNodes[cid] = clusterOptionsByData['clusterNodeProperties'];
	  // color_book[cid] = '#1E90FF';
	  network.fit();
  
  }
  // console.log(clustered)
}

function showAllComments(){
	html_cnt = "";
	for (var commentId in commentsDict){
		html_cnt += "<div class=\"block-item\">" + commentsDict[commentId] + "</div>";
		
	}
	$("#comments-box").html(html_cnt);
}

function rotate(cx, cy, x, y, angle) {
    var radians = (Math.PI / 180) * angle,
        cos = Math.cos(radians),
        sin = Math.sin(radians),
        nx = (cos * (x - cx)) + (sin * (y - cy)) + cx,
        ny = (cos * (y - cy)) - (sin * (x - cx)) + cy;
    return [nx, ny];
}

function hideAll(){
	html_cnt = "";
	for (var commentId in commentsDict){
		html_cnt += "<div class=\"block-item\">" + commentsDict[commentId] + "</div>";
	}
	$("#comments-box").html(html_cnt);
}

function expandCluster(clusterID){
	if (clusterID){
		network.openCluster(clusterID);
		// Remove node from dataset
		nodesDataset.remove(clusterID);
		delete allNodes[clusterID];
		cluster_ids.delete(clusterID);
		delete group_members[clusterID];
		// console.log(cluster_ids);
		if (cluster_ids.size == 0){
			clustered = false;
		$("#clusterButton").prop('value', 'Cluster nodes by topics'); 
		}
	}
}

function onFileSelected(event) {
	var selectedFile = event.target.files[0];
	var reader = new FileReader();
	reader.onload = function(event) {
	 jtext = event.target.result;
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
			n_comments = parsed_text.summary.n_comments;
			// Read the comments
			for (let item of parsed_text.comments){
				commentsDict[item.id] = item.label;
			}
			// console.log(commentsDict);
			draw();
			
			
		}
		catch(err) {
			console.log(err.message);
			alert("Unable to draw the network!", console.log(err.message));
		};
	 };	
  };

  reader.readAsText(selectedFile,"UTF-8");
}

function selectDesFile(){
	$( "#file-input" ).trigger( "click" );
}

function hideInspectWindow(){
	$('#inspect-network-window').toggle('slide'); $("#mynetwork").fadeIn('slow', function() { });
}

function draw_cluster(){
	var cluster_container = document.getElementById('myinspectnetwork');
	var cluster_options = {
	   autoResize: true,
	   height: '100%',
	   width: '100%',
	   locale: document.getElementById('locale').value,
	   clickToUse: false,
	   physics: {
		    enabled: true,
			barnesHut: {
			  centralGravity: 0.01,
			  springLength: 1,
			  springConstant: 0.01,
			  damping: 0.22,
			  avoidOverlap: 0.4,
			},
			maxVelocity: 20,
			minVelocity: .5,
			adaptiveTimestep: true,
			stabilization:{
				enabled: true,
				iterations: 1000,
			}
		  },
	   layout: {
	    improvedLayout:false,
	    randomSeed:seed
	  	},
	   
	   interaction:{
			hover:true,
			tooltipDelay: 200,
		},
	   nodes: {
		shape: 'square',
		font: {
				size: 28,
				color: '#ffffff'
			},
	     scaling: {
			  min: 20,
			  max: 40,
			  label: {
				min: 20,
				max: 40,
				drawThreshold: 12,
				maxVisible: 30
			  }
		  },  
	    },
	    edges: {
			arrows: {
			  to:     {enabled: true, scaleFactor:1, type:'arrow'},
			  middle: {enabled: false, scaleFactor:1, type:'arrow'},
			  from:   {enabled: false, scaleFactor:1, type:'arrow'}
			},
			color: {inherit: 'both'},
			smooth: {
			  enabled: true,
			  type: "dynamic",
			  roundness: 0.2
			},
			scaling:{
			  min: 1,
			  max: 10,
			  label: {
				enabled: true,
				min: 4,
				max: 20,
				maxVisible: 10,
				drawThreshold: 1
			  },
			  customScalingFunction: function (min,max,total,value) {
				if (max === min) {
				  return 0.5;
				}
				else {
				  var scale = 1 / (max - min);
				  return Math.max(0,(value - min)*scale);
				}
			  }
			},
	    },
	    configure:function (option, path) {
	      if (path.indexOf('smooth') !== -1 || option === 'smooth') {
	        return true;
	      }
	      return false;
	    },
	    groups: {
            central: {
                color: {background:'red',border:'white'},
                shape: 'star'
            },
            article: {
                color: {background:'red',border:'white'},
                shape: 'diamond'
            },
            comment: {
            color: {background:'red',border:'white'},
                shape: 'dot'
            }
        }
		  
	};	// End option
	var filtered_nodes = nodesDataset.get({
	  filter: function (item) {
		// console.log(filtered_nodes);
		return ($.inArray(item.id, group_members[selected_cluster].ids)>-1) && !network.isCluster(item.id);
	  }
	});
	// Recolor all nodes

	for (var i = 0; i < filtered_nodes.length; i++) {
	   filtered_nodes[i].color = color_book[filtered_nodes[i].id];
	}
	
	cluster_data = {
	  nodes: filtered_nodes,
	  edges: edgesDataset.get({
		  filter: function (item) {
			return (item.from == selected_cluster || item.to == selected_cluster || $.inArray(item.from, group_members[selected_cluster].ids)|| $.inArray(item.to, group_members[selected_cluster].ids));
		  }
		})
	  }; 
	// draw the network
	// console.log(cluster_data);
	var cluster_network = new vis.Network(cluster_container, cluster_data, cluster_options);
	cluster_network.on("hoverNode", function (params) {
        node_id = params.node;
        html_content = "";
		var related_edges = edgesDataset.get({
				filter: function (item) {
					return item.from == node_id && item.to.match(/^comment~[\d]+$/i);
			  }
		});
		
		for (let item of related_edges){
			html_content = html_content + "<div class=\"block-item\">" + commentsDict[item.to] + "</div>";
		}
		$("#comments-box").html(html_content); // Update to page
    });
	
}

$(document).ready(function () {
	drawFromJS();
});

$(function(){
  $.contextMenu({
    selector: '.context-menu-one', 
    trigger: 'none',
    callback: function(key, options) {
      var m = "clicked: " + key;
      if (key == 'fit'){ network.fit();}
      else if (key == 'expand'){expandCluster(selected_nodeid);}
      else if (key == 'inspect'){
		 $( "#mynetwork" ).fadeOut( "slow", function() { });
		 $( "#inspect-network-window" ).toggle( "slide" );
		 draw_cluster();
	  }
      // window.console && console.log(options) || alert(m);
      // console.log(selected_nodeid);

    },
    items: {
	  "expand"      : { name: "Expand Cluster",               icon: "expand-cluster" },
	  "inspect"      : { name: "Inspect Cluster",               icon: "inspect-cluster" },
	  "sep1"     : "---------",
	  "fit"      : { name: "Fit Graph",               
					 icon: "fit" 
					}
      // "cut"      : { name: "Cut",               icon: "cut" },
      // "copy"     : { name: "Copy",              icon: "copy" },
      // "paste"    : { name: "Paste",             icon: "paste" },
      // "sep1"     : "---------",
      // "google"   : { name: "Share on Google+",  icon: "google-plus" },
      // "facebook" : { name: "Share on Facebook", icon: "facebook" },
      // "sep2"     : "---------",
      // "save"     : { name: "Save",              icon: "save" },
      // "quit"     : { name: "Quit",              icon: "quit" }
    }
  });
});


