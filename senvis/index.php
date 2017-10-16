<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"
	"http://www.w3.org/TR/html4/loose.dtd">
<html>
	<head>
		<meta charset="utf-8">
		<title>Text Summarization visualization</title>
<!---		<script src="jquery-1.11.3.min.js"></script> 
		<script src="jquery-ui.js"></script>
		<script type="text/javascript" src="vis.js"></script>
		<link href="vis-network.min.css" rel="stylesheet" type="text/css"/> 
		--->
		<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.3/jquery.min.js"></script>
		<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery-contextmenu/2.6.2/jquery.contextMenu.min.js"></script>
		<link rel="stylesheet" href="https://ajax.googleapis.com/ajax/libs/jqueryui/1.11.3/themes/smoothness/jquery-ui.css">
		<script src="https://ajax.googleapis.com/ajax/libs/jqueryui/1.11.3/jquery-ui.min.js"></script>
		<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.20.1/vis.min.css">
		<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/jquery-contextmenu/2.6.2/jquery.contextMenu.min.css">
		<link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
		<link href="mystyle.css" rel="stylesheet" type="text/css"/> 
		<link href="vis.min.css" rel="stylesheet" type="text/css"/> 
		<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.20.1/vis.min.js"></script>
		<script src="https://cdnjs.cloudflare.com/ajax/libs/chroma-js/1.3.4/chroma.min.js"></script>
		<script type="text/javascript" src="exampleUtil.js"></script>
		<link rel="icon" 
		  type="image/png" 
		  href="img/logowhite.png">
		
		
	</head>
	<body onload="init();">
		<div id="netcontainer" > 
			<p style="margin-bottom:0.7vh;margin-top:0.7vh; margin-left:3vw; font-size: 0.2em;">
			  <!-- <label for="locale" style="font-family: Sans-serif, serif; font-size: 12px;">Select a locale:</label> -->
			  <select id="locale" onchange="drawFromJS();">
			    <option value="en">English</option>
			    <option value="de">Deutsch</option>
			    <option value="es">Español</option>
			    <option value="it">Italiano</option>
			    <option value="nl">Nederlands</option>
			    <option value="pt-br">Português</option>
			    <option value="ru">Русский </option>
			  </select>
			</p>
			<div id="node-popUp">
			  <span id="node-operation">node</span> <br>
			  <table style="margin:auto;">
			    <tr>
			      <td>id</td><td><input id="node-id" value="new id" /></td>
			    </tr>
			    <tr>
			      <td>label</td><td><input id="node-label" value="new value" /></td>
			    </tr>
			    <tr>
			      <td>*color*</td><td><input id="node-color" value="#3498DB" /></td>
			    </tr>
			    <tr>
			      <td>*value*</td><td><input id="node-value" value=1 /></td>
			    </tr>
			  </table>
			  <input type="button" value="save" id="node-saveButton" class="flatbutton" style="width:7vw;height: 4vh;"/>
			  <input type="button" value="cancel" id="node-cancelButton" class="flatbutton" style="width:7vw;height: 4vh;"/>
			</div>
			
			<div id="edge-popUp">
			  <span id="edge-operation">edge</span> <br>
			  <table style="margin:auto;">
			    <tr>
			      <td>label</td><td><input id="edge-label" value="new label" /></td>
			    </tr>
			    <tr>
			      <td>value</td><td><input id="edge-value" value="new value" /></td>
			    </tr> 
			  </table>
			  <input type="button" value="save" id="edge-saveButton" class="flatbutton" style="width:7vw;height: 4vh;"/>
			  <input type="button" value="cancel" id="edge-cancelButton" class="flatbutton" style="width:7vw;height: 4vh;"/>
			</div>
			<div id="mynetwork" class="context-menu-one"> 
				<div class="welcome w3-animate-opacity"><img src="img/sis.png" alt="SISLab's logo" style="position: absolute; top: 0; bottom:0; left: 0; right:0; margin: auto;"></div>
			</div> 
			<!-- Inspection panel and button --> 
			<div id="inspect-network-window" style="display:none;" class=""> 
				<div id="myinspectnetwork" class=""></div> 
				<input type="button" value="" id="close-inspect-button" class="round-button" style="width: 2.5vw;height: 2.5vw; left: 72vw;" onclick="hideInspectWindow();"/>
			</div>
			
			
			
			<div id="comments-box"> 
				<div class="welcome flashit" style="position: absolute; top: 0; bottom:0; left: 0; right:0; margin: auto;">WELCOME</div>
			</div> 
			<div id = "grabar" class= "vertical-text">
				<div class= "textcentering">-<br>~<br>+</div>
			</div> 
		</div>
		<div id = "commandbox">
			<!-- <textarea id="jsonarea" onclick="this.select()">Drop graph description JSON file or paste its content here...</textarea>  
			<button id = "submitbutton" class="flatbutton" style="position:absolute; width:9vw;	height: 6vh;">Draw</button>  -->
			<input type="button" onclick="selectDesFile();" value="Open & Draw" id="Openfile" class="open-icon haftSizeButton"> 
			<input type="button" onclick="" value="Redraw" id="submitbutton" class="draw-icon haftSizeButton" disabled> 
			<input type="button" onclick="handleExpandCluster()" value="Expand all clusters" id="clusterButton" class="cluster-icon haftSizeButton" disabled> 
			<input id="file-input" type="file" name="name" style="display: none;"  accept=".json" onchange="onFileSelected(event)"/>
			<div  style="right: 0vw; top: 0.0vw; width: 20vw; position: absolute; ">
				<label class="checkbox-container">Hide nodes history
				  <input type="checkbox" checked="checked" id="his-history-chk" onclick="onHistoryShowHideChange(event);">
				  <span class="checkmark"></span>
				</label>
				<label class="checkbox-container">Enable fast stabilization
				  <input type="checkbox" checked="checked" id="fast-stabilization-chk" onclick="onChangeStabilizationSpeed(event);">
				  <span class="checkmark"></span>
				</label>
			</div>
			<p id="status">File API & FileReader API not supported</p>
			
		</div>
		<script>
		  function handleFileSelect(evt) {
			evt.stopPropagation();
			evt.preventDefault();

			var files = evt.dataTransfer.files; // FileList object.
			var reader = new FileReader();  
			reader.onload = function(event) {            
				 // document.getElementById('jsonarea').value = event.target.result;
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
							// console.log(commentsDict);
							commentsDict[item.id] = {label: item.label, user: item.user, time: item.time, sen: item.sen_score};
						}
						// console.log(commentsDict);
						draw();
						handleExpandCluster();
					}
					catch(err) {
						console.log(err.message);
						alert("Unable to draw the network!", console.log(err.message));
					};
				 };	
			}        
			reader.readAsText(files[0],"UTF-8");
		  }

		  function handleDragOver(evt) {
			evt.stopPropagation();
			evt.preventDefault();
			evt.dataTransfer.dropEffect = 'copy'; // Explicitly show this is a copy.
		  }

		  // Setup the dnd listeners.
		  // var dropZone = document.getElementById('jsonarea');
		  dropZone = document.getElementById('mynetwork');
		  dropZone.addEventListener('dragover', handleDragOver, false);
		  dropZone.addEventListener('drop', handleFileSelect, false);
		</script>
		<script src="myjavascript.js"></script>
	</body>
</html>
