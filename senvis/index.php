<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"
	"http://www.w3.org/TR/html4/loose.dtd">
<html>
	<head>
		<meta charset="utf-8">
		<title>Text Summarization visualization</title>
		<script src="jquery-1.11.3.min.js"></script>
		<script type="text/javascript" src="exampleUtil.js"></script>
		<script type="text/javascript" src="vis.js"></script>
		<link href="vis-network.min.css" rel="stylesheet" type="text/css"/> 
		<link href="mystyle.css" rel="stylesheet" type="text/css"/> 
		<script src="jquery-ui.js"></script>
		
	</head>
	<body onload="init();">
		<div id="netcontainer" > 
			<p style="margin-bottom:0.7vh;margin-top:0.7vh;">
			  <label for="locale" style="font-family: Sans-serif, serif; font-size: 12px;">Select a locale:</label>
			  <select id="locale" onchange="drawFromJS();">
			    <option value="en">Enlish</option>
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
			<div id="mynetwork"> </div> 
			<div id = "grabar" class= "vertical-text">
				<div class= "textcentering">-<br>~<br>+</div>
			</div> 
		</div>
		<div id = "commandbox">
			<textarea id="jsonarea" onclick="this.select()">Paste graph description or enter JSON here</textarea>  
			<button id = "submitbutton" class="flatbutton" style="position:absolute; width:9vw;	height: 8vh;">Draw</button>
			<p id="status">File API & FileReader API not supported</p>

		</div>
		<script>
		  function handleFileSelect(evt) {
			evt.stopPropagation();
			evt.preventDefault();

			var files = evt.dataTransfer.files; // FileList object.
			var reader = new FileReader();  
			reader.onload = function(event) {            
				 document.getElementById('jsonarea').value = event.target.result;
			}        
			reader.readAsText(files[0],"UTF-8");
		  }

		  function handleDragOver(evt) {
			evt.stopPropagation();
			evt.preventDefault();
			evt.dataTransfer.dropEffect = 'copy'; // Explicitly show this is a copy.
		  }

		  // Setup the dnd listeners.
		  var dropZone = document.getElementById('jsonarea');
		  dropZone.addEventListener('dragover', handleDragOver, false);
		  dropZone.addEventListener('drop', handleFileSelect, false);
		</script>
		<script src="myjavascript.js"></script>
	</body>
</html>
