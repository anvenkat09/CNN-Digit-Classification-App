var canvas = document.querySelector(".canvas");
var context = canvas.getContext("2d");
var canvas_bounding_box = canvas.getBoundingClientRect();
var clear_button = document.querySelector(".clear");
var predict_button = document.querySelector(".predict");
var predicted_result_text = document.querySelector(".predicted_result")

var prev_pos_X = 0;
var prev_pos_Y = 0;
var curr_pos_X = 0;
var curr_pos_Y = 0;

var mouse_pressed = false;

//Mouse not pressed
const onMouseUp = (e) => {
	mouse_pressed = false;
}

//Mouse pressed down
const onMouseDown = (e) => {
	mouse_pressed = true;
	prev_pos_X = curr_pos_X;
	prev_pos_Y = curr_pos_Y;

	//find new mouse coordinates on canvas relative to scaling and sizing
	curr_pos_X = (e.clientX - canvas_bounding_box.left) / (canvas_bounding_box.right - canvas_bounding_box.left) * canvas.width;
	curr_pos_Y = (e.clientY - canvas_bounding_box.top) / (canvas_bounding_box.bottom - canvas_bounding_box.top) * canvas.height;
}

//Mouse moved outside canvas
const onMouseOut = (e) => {
	mouse_pressed = false;
}

//Mouse moving, draw path
const onMouseMove = (e) => {
	if(mouse_pressed){
		prev_pos_X = curr_pos_X;
		prev_pos_Y = curr_pos_Y;

		//find new mouse coordinates on canvas relative to scaling and sizing
		curr_pos_X = (e.clientX - canvas_bounding_box.left) / (canvas_bounding_box.right - canvas_bounding_box.left) * canvas.width;
		curr_pos_Y = (e.clientY - canvas_bounding_box.top) / (canvas_bounding_box.bottom - canvas_bounding_box.top) * canvas.height;

		//draw path across mouse position
		context.beginPath();
		context.lineCap = 'round';
		context.moveTo(prev_pos_X, prev_pos_Y);
		context.lineTo(curr_pos_X, curr_pos_Y);
		context.strokeStyle = "#000000";
		context.lineWidth = 10;
		context.stroke();
		context.closePath();
	}
}

const onClear = (e) => {
	context.clearRect(0,0, canvas.width, canvas.height);
	predicted_result_text.innerHTML = ""
}

const onPredict = (e) => {
	var image_url_base64 = canvas.toDataURL();
	$.ajax({
		type: "POST",
		url: "/predict",
		data:{
			imageBase64: image_url_base64
		},
		success: function(response){
			$(predicted_result_text).text("Predicted Result: " + response);
		}
	});
}

document.addEventListener("mouseout", onMouseOut);
document.addEventListener("mousemove", onMouseMove);
document.addEventListener("mouseup", onMouseUp);
document.addEventListener("mousedown", onMouseDown);
clear_button.addEventListener("click", onClear);
predict_button.addEventListener("click", onPredict);