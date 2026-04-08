from ingestion.rtsp_client import RTSPClient
from ingestion.frame_buffer import FrameBuffer
from ingestion.clip_assembler import ClipAssembler
from pose.rtmpose_model import RTMPoseModel
from activity.slowfast_inference import SlowFastInference

# Sample video or RTSP
video_path = "data/sample_videos/factory_clip.mp4"

# Setup pipeline
client = RTSPClient(video_path)
buffer = FrameBuffer(max_size=60)
assembler = ClipAssembler(clip_length=16, stride=8)
pose_model = RTMPoseModel()
activity_model = SlowFastInference()

for frame in client:
    buffer.add(frame)
    clip = assembler.get_clip(buffer)
    if clip is not None:
        keypoints = pose_model.infer(clip)
        activity = activity_model.infer(clip, keypoints)
        print(activity)