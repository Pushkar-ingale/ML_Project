
import React, { useEffect, useRef, useState } from "react";
import {
  Box,
  Typography,
  Card,
  CardContent,
  Alert,
  AlertTitle,
  CircularProgress,
} from "@mui/material";

const MarkAttendance = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [alert, setAlert] = useState(null);
  const [processing, setProcessing] = useState(false);

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error("Error occurred while accessing Camera: ", error);
        setAlert({
          type: "error",
          title: "Camera Error",
          message: "Failed to access the camera",
        });
      }
    };
    startCamera();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      if (!processing) {
        captureAndSendFrame();
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [processing]);

  const captureAndSendFrame = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");

    context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

    const imageBase64 = canvas.toDataURL("image/jpeg");
    sendImageToBackend(imageBase64);
  };

  const sendImageToBackend = async (imageBase64) => {
    setProcessing(true);

    try {
      const response = await fetch("http://127.0.0.1:5000/process_frame", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: imageBase64 }),
      });

      const result = await response.json();

      if (result.success) {
        setAlert({
          type: "success",
          title: "Attendance Marked",
          message: `Welcome, ${result.name} (PRN: ${result.prnno})!`,
        });
      } else {
        setAlert({
          type: "error",
          title: "Face Not Recognized",
          message: "Please try again",
        });
      }
    } catch (error) {
      console.error("Error occurred while sending Images:", error);
      setAlert({
        type: "error",
        title: "Error",
        message: "Failed to mark Attendance",
      });
    }
    setProcessing(false);
  };

  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        minHeight: "100vh",
        backgroundColor: "#f4f4f9",
        padding: 2,
      }}
    >
      <Card sx={{ maxWidth: 700, width: "100%", boxShadow: 3 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom align="center">
            Real Time Attendance System
          </Typography>

          {alert && (
            <Alert severity={alert.type} sx={{ mb: 2 }}>
              <AlertTitle>{alert.title}</AlertTitle>
              {alert.message}
            </Alert>
          )}

          {processing && (
            <Box sx={{ display: "flex", justifyContent: "center", mb: 2 }}>
              <CircularProgress />
            </Box>
          )}

          <Box sx={{ display: "flex", justifyContent: "center", mb: 2 }}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              width="640"
              height="480"
              style={{ borderRadius: "10px", border: "2px solid #ccc" }}
            />
          </Box>

          <canvas
            ref={canvasRef}
            width="640"
            height="480"
            style={{ display: "none" }}
          />
        </CardContent>
      </Card>
    </Box>
  );
};

export default MarkAttendance;

