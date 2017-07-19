<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"
	"http://www.w3.org/TR/html4/loose.dtd">
<html>
	<head>
		<title></title>
		<script src="jquery-1.11.3.min.js"></script>
		<script type="text/javascript" src="vis.js"></script>
		<!-- <link href="vis.css" rel="stylesheet" type="text/css" />-->
		<link href="vis-network.min.css" rel="stylesheet" type="text/css"/> 
		<style type="text/css">
			.newspaper {
				-webkit-column-count: 3; /* Chrome, Safari, Opera */
				-moz-column-count: 3; /* Firefox */
				column-count: 3;
			}
			#status {
					visibility: hidden;
			}
			#mynetwork {
				position:absolute
				width: 80vw;
				height: 85vh;
				border: 1px solid #444444;
				background-color: #222222;
			}
			
			#text {
				position:absolute;
				top:8px;
				left:530px;
				width:30px;
				height:50px;
				margin:auto auto auto auto;
				font-size:22px;
				color: #000000;
			}


			div.outerBorder {
				position:relative;
				top:400px;
				width:600px;
				height:44px;
				margin:auto auto auto auto;
				border:8px solid rgba(0,0,0,0.1);
				background: rgb(252,252,252); /* Old browsers */
				background: -moz-linear-gradient(top,  rgba(252,252,252,1) 0%, rgba(237,237,237,1) 100%); /* FF3.6+ */
				background: -webkit-gradient(linear, left top, left bottom, color-stop(0%,rgba(252,252,252,1)), color-stop(100%,rgba(237,237,237,1))); /* Chrome,Safari4+ */
				background: -webkit-linear-gradient(top,  rgba(252,252,252,1) 0%,rgba(237,237,237,1) 100%); /* Chrome10+,Safari5.1+ */
				background: -o-linear-gradient(top,  rgba(252,252,252,1) 0%,rgba(237,237,237,1) 100%); /* Opera 11.10+ */
				background: -ms-linear-gradient(top,  rgba(252,252,252,1) 0%,rgba(237,237,237,1) 100%); /* IE10+ */
				background: linear-gradient(to bottom,  rgba(252,252,252,1) 0%,rgba(237,237,237,1) 100%); /* W3C */
				filter: progid:DXImageTransform.Microsoft.gradient( startColorstr='#fcfcfc', endColorstr='#ededed',GradientType=0 ); /* IE6-9 */
				border-radius:72px;
				box-shadow: 0px 0px 10px rgba(0,0,0,0.2);
			}

			#border {
				position:absolute;
				top:10px;
				left:10px;
				width:500px;
				height:23px;
				margin:auto auto auto auto;
				box-shadow: 0px 0px 4px rgba(0,0,0,0.2);
				border-radius:10px;
			}

			#bar {
				position:absolute;
				top:0px;
				left:0px;
				width:20px;
				height:20px;
				margin:auto auto auto auto;
				border-radius:11px;
				border:2px solid rgba(30,30,30,0.05);
				background: rgb(0, 173, 246); /* Old browsers */
				box-shadow: 2px 0px 4px rgba(0,0,0,0.4);
			}
			
			#commandbox{
				position:absolute;
				top: 87vh;
				left -20px;
				width:98vw;
				height:9.5vh;
				margin:bottom;
				border:8px solid rgba(0,0,0,0.1);
				box-shadow: 0px 0px 10px rgba(0,0,0,0.2);
			}
			
			#grabar{
				position:absolute;
				top: 9.5px; 
				left:97.6vw;
				width:1.7vw;
				height: 85vh;
				background: red; /* For browsers that do not support gradients */
				background: linear-gradient(red, #ff0a00, #ff1500, #ff1f00, #ff2900, #ff3400, #ff3e00, #ff4800, #ff5200, #ff5d00, #ff6700, #ff7100, #ff7c00, #ff8600, #ff9000, #ff9b00, orange, #ffaf00, #ffb900, #ffc400, #ffce00, #ffd800, #ffe300, #ffed00, #fff700, #fcff00, #f2ff00, #e8ff00, #deff00, #d3ff00, #c9ff00, #bfff00, #b4ff00, #af0, #a0ff00, #95ff00, #8bff00, #81ff00, #76ff00, #6cff00, #62ff00, #58ff00, #4dff00, #43ff00, #39ff00, #2eff00, #24ff00, #1aff00, #0fff00, #05ff00, #00ff05, #00ff0f, #00ff1a, #00ff24, #00ff2e, #00ff39, #00ff43, #00ff4d, #00ff58, #00ff62, #00ff6c, #00ff76, #00ff81, #00ff8b, #00ff95, #00ffa0, #0fa, #00ffb4, #00ffbf, #00ffc9, #00ffd3, #00ffde, #00ffe8, #00fff2, #00fffc, #00f7ff, #00edff, #00e3ff, #00d8ff, #00ceff, #00c4ff, #00b9ff, #00afff, #00a5ff, #009bff, #0090ff, #0086ff, #007cff, #0071ff, #0067ff, #005dff, #0052ff, #0048ff, #003eff, #0034ff, #0029ff, #001fff, #0015ff, #000aff, blue); /* Standard syntax */
				text-align: center;
				vertical-align: middle;
				margin:auto;
			}
				
			#jsonarea{
				position:absolute;
				left:10px;
				top:5px;
				width: 87vw;
				height: 8vh;
			}
			
			.textcentering{
				position: relative;
				float: left;
				top: 50%;
				left: 50%;
				transform: translate(-50%, -50%);
			}
			
			#netcontainer{
			}
			.flatbutton {
				position:absolute;
				color: #fff;
				background-color: #6496c8;
				text-shadow: -1px 1px #417cb8;
				border: none;
				left:88.5vw;
				top:5px;
				width:9vw;
				height: 8vh;
				}

			.flatbutton:hover,
				section.flat button.hover {
				  background-color: #346392;
				  text-shadow: -1px 1px #27496d;
				}

			.flatbutton:active,
			.flatbutton.active {
				  background-color: #27496d;
				  text-shadow: -1px 1px #193047;
				}
			
			.div.vis-configuration-wrapper {
				visibility: hidden;
			}	
			
			.floatBox{
				position: fixed;
				float: right;
				color: #F2F4F4;
				margin-right: 25px;
				margin-bottom: 50px;
				top: 50px;
			}
		</style>
		<script src="jquery-ui.js"></script>
		
	</head>
	<body>
		<div id="netcontainer" > 
			<div id="mynetwork"> </div> 
			<div id = "grabar" class= "vertical-text">
				<div class= "textcentering">-<br>~<br>+</div>
			</div> 
		</div>
		<div id = "commandbox">
			<textarea id="jsonarea" onclick="this.select()">Paste graph description or enter JSON here</textarea>  
			<button id = "submitbutton" class="flatbutton">Draw</button>
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
