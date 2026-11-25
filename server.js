
const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
const multer = require("multer");
const rateLimit = require("express-rate-limit");
const admin = require("firebase-admin");
const { GoogleGenAI } = require("@google/genai");
const { Storage } = require("@google-cloud/storage");
const crypto = require("crypto");

const app = express();
const PORT = process.env.PORT || 10000;

app.use(helmet());
app.use(cors({
  origin: process.env.CORS_ORIGIN
    ? process.env.CORS_ORIGIN.split(",")
    : true
}));
app.use(express.json({ limit: "2mb" }));

app.use(rateLimit({
  windowMs: 15 * 60 * 1000,
  limit: 100
}));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowed = ["image/jpeg", "image/png"];
    if (!allowed.includes(file.mimetype)) {
      return cb(new Error("Only JPG and PNG images are allowed"));
    }
    cb(null, true);
  }
});

// In-memory job store.
// Note: jobs are lost if the Render instance restarts.
const jobs = new Map();

let firebaseReady = false;
let aiClient = null;
let storageClient = null;

// Firebase Authentication
try {
  if (process.env.FIREBASE_SERVICE_ACCOUNT_JSON) {
    const firebaseKey = JSON.parse(
      process.env.FIREBASE_SERVICE_ACCOUNT_JSON
    );

    admin.initializeApp({
      credential: admin.credential.cert(firebaseKey)
    });

    firebaseReady = true;
  }
} catch (error) {
  console.error("Firebase setup error:", error.message);
}

// Google Vertex AI / Gemini Veo
function getAI() {
  if (aiClient) return aiClient;

  const project = process.env.GOOGLE_CLOUD_PROJECT;
  const credentialsText = process.env.GOOGLE_SERVICE_ACCOUNT_JSON;

  if (!project || !credentialsText) {
    throw new Error("Google Cloud credentials are not configured");
  }

  const credentials = JSON.parse(credentialsText);

  aiClient = new GoogleGenAI({
    enterprise: true,
    project,
    location: process.env.GOOGLE_CLOUD_LOCATION || "global",
    googleAuthOptions: {
      credentials,
      scopes: [
        "https://www.googleapis.com/auth/cloud-platform"
      ]
    }
  });

  return aiClient;
}

// Google Cloud Storage
function getStorage() {
  if (storageClient) return storageClient;

  const projectId = process.env.GOOGLE_CLOUD_PROJECT;
  const credentials = JSON.parse(
    process.env.GOOGLE_SERVICE_ACCOUNT_JSON || "{}"
  );

  storageClient = new Storage({
    projectId,
    credentials
  });

  return storageClient;
}

// Firebase Bearer token verification
async function authenticate(req, res, next) {
  if (!firebaseReady) {
    return res.status(503).json({
      success: false,
      error: "Firebase authentication is not configured"
    });
  }

  const header = req.headers.authorization || "";
  const token = header.startsWith("Bearer ")
    ? header.substring(7)
    : "";

  if (!token) {
    return res.status(401).json({
      success: false,
      error: "Firebase Bearer ID token required"
    });
  }

  try {
    const decoded = await admin.auth().verifyIdToken(token);
    req.user = decoded;
    next();
  } catch (error) {
    return res.status(401).json({
      success: false,
      error: "Invalid or expired Firebase ID token"
    });
  }
}

function validateOptions(body = {}) {
  const aspectRatio = body.aspectRatio || "9:16";

  if (!["9:16", "16:9"].includes(aspectRatio)) {
    throw new Error("aspectRatio must be 9:16 or 16:9");
  }

  // Veo standard generation duration.
  // 15-second generation is not supported by this configuration.
  return {
    aspectRatio,
    durationSeconds: 8,
    resolution: "1080p"
  };
}

// Create video generation job
async function createVideoJob({
  userId,
  prompt,
  image
}) {
  if (!prompt || typeof prompt !== "string" || prompt.length > 4000) {
    throw new Error("A valid prompt (maximum 4000 characters) is required");
  }

  const bucketName = process.env.GCS_BUCKET;
  if (!bucketName) {
    throw new Error("GCS_BUCKET is not configured");
  }

  const options = validateOptions(
    arguments[0].options || {}
  );

  const jobId = crypto.randomUUID();
  const ai = getAI();

  const source = {
    prompt: prompt.trim()
  };

  if (image) {
    source.image = {
      imageBytes: image.buffer.toString("base64"),
      mimeType: image.mimetype
    };
  }

  const operation = await ai.models.generateVideos({
    model: process.env.VEO_MODEL || "veo-3.1-fast-generate-001",
    source,
    config: {
      numberOfVideos: 1,
      aspectRatio: options.aspectRatio,
      durationSeconds: options.durationSeconds,
      resolution: options.resolution,
      personGeneration: "allow_adult",
      enhancePrompt: true,
      outputGcsUri: `gs://${bucketName}/vivek-ai/${jobId}/`
    }
  });

  jobs.set(jobId, {
    jobId,
    userId,
    operation,
    status: operation.done ? "processing" : "processing",
    videoUri: null,
    createdAt: new Date().toISOString(),
    error: null
  });

  return jobId;
}

// Health check
app.get("/api/v1/health", (req, res) => {
  res.json({
    success: true,
    app: "Vivek AI Studio",
    backend: "online",
    provider: "Google Gemini Veo",
    version: "1.0.0"
  });
});

// Text to Video
app.post(
  "/api/v1/videos/text-to-video",
  authenticate,
  async (req, res) => {
    try {
      const { prompt } = req.body || {};
      const options = validateOptions(req.body || {});

      const jobId = await createVideoJob({
        userId: req.user.uid,
        prompt,
        options
      });

      res.status(202).json({
        success: true,
        jobId,
        status: "processing",
        message: "Veo video generation started",
        durationSeconds: 8,
        resolution: "1080p"
      });
    } catch (error) {
      console.error("Text video error:", error.message);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  }
);

// Photo to Video
app.post(
  "/api/v1/videos/photo-to-video",
  authenticate,
  upload.fields([
    { name: "image", maxCount: 1 },
    { name: "photo", maxCount: 1 }
  ]),
  async (req, res) => {
    try {
      const files = req.files || {};
      const image = files.image?.[0] || files.photo?.[0];

      if (!image) {
        return res.status(400).json({
          success: false,
          error: "Upload an image using image or photo field"
        });
      }

      const prompt = req.body.prompt ||
        "Animate this image naturally with realistic cinematic camera movement.";

      const options = validateOptions(req.body);

      const jobId = await createVideoJob({
        userId: req.user.uid,
        prompt,
        image,
        options
      });

      res.status(202).json({
        success: true,
        jobId,
        status: "processing",
        message: "Photo to video generation started",
        durationSeconds: 8,
        resolution: "1080p"
      });
    } catch (error) {
      console.error("Photo video error:", error.message);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  }
);

// Check job status
app.get(
  "/api/v1/jobs/:jobId",
  authenticate,
  async (req, res) => {
    try {
      const job = jobs.get(req.params.jobId);

      if (!job || job.userId !== req.user.uid) {
        return res.status(404).json({
          success: false,
          error: "Job not found"
        });
      }

      if (!job.videoUri && job.status !== "failed") {
        const ai = getAI();

        const operation = await ai.operations.get({
          operation: job.operation
        });

        job.operation = operation;

        if (operation.done) {
          if (operation.error) {
            job.status = "failed";
            job.error = operation.error.message || "Video generation failed";
          } else {
            const video =
              operation.response?.generatedVideos?.[0]?.video;

            if (video?.uri) {
              job.videoUri = video.uri;
              job.status = "completed";
            } else {
              job.status = "failed";
              job.error = "Generated video URI was not returned";
            }
          }
        }
      }

      res.json({
        success: true,
        jobId: job.jobId,
        status: job.status,
        videoReady: Boolean(job.videoUri),
        error: job.error
      });
    } catch (error) {
      console.error("Job status error:", error.message);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  }
);

// Download generated MP4
app.get(
  "/api/v1/videos/:jobId/download",
  authenticate,
  async (req, res) => {
    try {
      const job = jobs.get(req.params.jobId);

      if (!job || job.userId !== req.user.uid) {
        return res.status(404).json({
          success: false,
          error: "Job not found"
        });
      }

      if (job.status !== "completed" || !job.videoUri) {
        return res.status(409).json({
          success: false,
          error: "Video is not ready yet"
        });
      }

      const match = /^gs:\/\/([^/]+)\/(.+)$/.exec(job.videoUri);

      if (!match || match[1] !== process.env.GCS_BUCKET) {
        return res.status(500).json({
          success: false,
          error: "Invalid generated video storage URI"
        });
      }

      const bucket = getStorage().bucket(match[1]);
      const file = bucket.file(match[2]);

      const [exists] = await file.exists();

      if (!exists) {
        return res.status(404).json({
          success: false,
          error: "Generated MP4 file not found in storage"
        });
      }

      res.setHeader("Content-Type", "video/mp4");
      res.setHeader(
        "Content-Disposition",
        `attachment; filename="vivek-ai-${job.jobId}.mp4"`
      );

      file.createReadStream()
        .on("error", (error) => {
          console.error("Download stream error:", error.message);
          if (!res.headersSent) res.status(500).end();
          else res.destroy(error);
        })
        .pipe(res);

    } catch (error) {
      console.error("Download error:", error.message);
      if (!res.headersSent) {
        res.status(500).json({
          success: false,
          error: error.message
        });
      }
    }
  }
);

// General error handler
app.use((error, req, res, next) => {
  console.error("Server error:", error.message);

  if (error instanceof multer.MulterError) {
    return res.status(400).json({
      success: false,
      error: error.message
    });
  }

  res.status(400).json({
    success: false,
    error: error.message || "Request failed"
  });
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Vivek AI Studio backend running on port ${PORT}`);
});

