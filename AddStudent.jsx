import React, { useState, useRef } from "react";
import {
  Box,
  Button,
  TextField,
  Typography,
  Card,
  CardContent,
  Alert,
  AlertTitle,
  LinearProgress,
} from "@mui/material";
import Background from "../assets/WhiteBackground.jpg"

const AddStudent = () => {
  const [prnno, setPrnno] = useState("");
  const [name, setName] = useState("");
  const [images, setImages] = useState([]);
  const [alert, setAlert] = useState(null);
  const [progress, setProgress] = useState(0); // Track progress
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  let captureInterval;

  const startCameraAndCapture = async () => {
    if (!prnno || !name) {
      setAlert({
        type: "error",
        title: "Error",
        message: "Please fill in all fields.",
      });
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) videoRef.current.srcObject = stream;

      const capturedImages = [];
      let count = 0;

      captureInterval = setInterval(() => {
        if (count >= 30) {
          clearInterval(captureInterval);
          stream.getTracks().forEach((track) => track.stop());
          submitData(prnno, name, capturedImages);
          return;
        }

        if (canvasRef.current && videoRef.current) {
          const canvas = canvasRef.current;
          const video = videoRef.current;
          const context = canvas.getContext("2d");
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          context.drawImage(video, 0, 0, canvas.width, canvas.height);

          const imageData = canvas.toDataURL("image/jpeg");
          capturedImages.push(imageData);
          setProgress(((count + 1) / 30) * 100); // Update progress
        }

        count++;
      }, 1000);
      setImages(capturedImages);
    } catch (error) {
      setAlert({
        type: "error",
        title: "Error",
        message: "Error accessing the camera. Please check permissions.",
      });
      console.error(error);
    }
  };

  const submitData = async (prnno, name, images) => {
    try {
      const response = await fetch("http://127.0.0.1:5000/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prnno, name, images }),
      });

      if (response.ok) {
        setAlert({
          type: "success",
          title: "Success",
          message: "Data submitted successfully!",
        });
        setPrnno(""); // Reset fields
        setName("");
        setImages([]);
        setProgress(0); // Reset progress
      } else {
        setAlert({
          type: "error",
          title: "Error",
          message: "Failed to submit data.",
        });
      }
    } catch (error) {
      setAlert({
        type: "error",
        title: "Error",
        message: "Error submitting data. Please try again later.",
      });
      console.error("Error submitting data:", error);
    }
  };

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100vh",
        backgroundColor: "#f4f4f9",
        padding: 2,
        backgroundImage: `url(${Background})`, // Use template literals to embed the variable
      backgroundSize: "cover", // Optional: adjust the size of the image
      backgroundPosition: "center", // Optional: align the image
      backgroundRepeat: "no-repeat", 
      }}
    >
      <Typography variant="h4" gutterBottom color="primary">
        Add Student
      </Typography>

      {alert && (
        <Alert
          severity={alert.type}
          sx={{ marginBottom: 2, width: "100%", maxWidth: 400 }}
        >
          <AlertTitle>{alert.title}</AlertTitle>
          {alert.message}
        </Alert>
      )}

      <Card sx={{ width: 400, padding: 3, boxShadow: 3 }}>
        <CardContent>
          <form
            onSubmit={(e) => e.preventDefault()}
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "15px",
            }}
          >
            <TextField
              label="PRN Number"
              variant="outlined"
              fullWidth
              required
              value={prnno}
              onChange={(e) => setPrnno(e.target.value)}
            />
            <TextField
              label="Name"
              variant="outlined"
              fullWidth
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
            <Button
              variant="contained"
              color="success"
              onClick={startCameraAndCapture}
              sx={{
                fontWeight: "bold",
                transition: "background-color 0.3s ease",
                ":hover": {
                  backgroundColor: "green",
                },
              }}
            >
              Start Camera & Submit
            </Button>
          </form>
        </CardContent>
      </Card>

      {progress > 0 && (
        <Box sx={{ width: "100%", maxWidth: 400, marginTop: 2 }}>
          <Typography variant="body2" color="textSecondary">
            Capturing Images ({Math.round(progress)}%)
          </Typography>
          <LinearProgress variant="determinate" value={progress} />
        </Box>
      )}

      <Box
        sx={{
          marginTop: 3,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        <video
          ref={videoRef}
          autoPlay
          style={{
            width: "300px",
            height: "300px",
            border: "1px solid #ccc",
            marginTop: "10px",
          }}
        />
        <canvas ref={canvasRef} style={{ display: "none" }} />
      </Box>
    </Box>
  );
};

export default AddStudent;