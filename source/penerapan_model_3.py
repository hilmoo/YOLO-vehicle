# Ultralytics YOLO 🚀, AGPL-3.0 license
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

track_history = defaultdict(list)

weights = "a/best.pt"
source = "video_3.mp4"
output = "penerapan_model_3.mp4"
device = "0"
out1_list = []
out2_list = []
out3_list = []
out4_list = []
in1_list = in2_list = in3_list = in4_list = []
current_region = None
in_region = [
    {
        "name": "kiri atas",
        "in1": Polygon([(229, 336), (229, 197), (364, 160), (364, 338)]),
        "counts1": 0,
        "counts2": 0,
        "counts3": 0,
        "counts4": 0,
        "dragging": False,
        "region_color": (255, 255, 225),  # BGR Value
    },
    {
        "name": "atas kanan",
        "in1": Polygon([(700, 117), (1031, 115), (912, -50), (702, -45)]),
        "counts1": 0,
        "counts2": 0,
        "counts3": 0,
        "counts4": 0,
        "dragging": False,
        "region_color": (255, 255, 225),  # BGR Value
    },
    {
        "name": "kanan bawah",
        "in1": Polygon([(853, 428), (850, 550), (1208, 617), (1203, 427)]),
        "counts1": 0,
        "counts2": 0,
        "counts3": 0,
        "counts4": 0,
        "dragging": False,
        "region_color": (255, 255, 225),  # BGR Value
    },
    {
        "name": "bawah kiri",
        "in1": Polygon([(614, 734), (405, 734), (290, 596), (614, 601)]),
        "counts1": 0,
        "counts2": 0,
        "counts3": 0,
        "counts4": 0,
        "dragging": False,
        "region_color": (255, 255, 225),  # BGR Value
    },
]
out_region = [
    {
        "name": "kanan atas",
        "out1": Polygon([(1293, 173), (1301, 417), (871, 409), (881, 151)]),
        "counts": 0,
        "dragging": False,
        "region_color": (255, 0, 225),  # BGR Value
        "text_color": (0, 0, 0),  # Region Text Color
    },
    {
        "name": "bawah kanan",
        "out1": Polygon([(630, 717), (861, 715), (992, 600), (632, 600)]),
        "counts": 0,
        "dragging": False,
        "region_color": (245, 76, 79),  # BGR Value
        "text_color": (0, 0, 0),  # Region Text Color
    },
    {
        "name": "kiri bawah",
        "out1": Polygon([(29, 586), (29, 347), (364, 347), (364, 588)]),
        "counts": 0,
        "dragging": False,
        "region_color": (25, 110, 11),  # BGR Value
        "text_color": (0, 0, 0),  # Region Text Color
    },
    {
        "name": "atas kiri",
        "out1": Polygon([(400, 117), (670, 115), (670, -50), (402, -45)]),
        "counts": 0,
        "dragging": False,
        "region_color": (41, 122, 128),  # BGR Value
        "text_color": (0, 0, 0),  # Region Text Color
    },
]


def run(
    weights=weights,
    source=source,
    output=output,
    device=device,
    view_img=True,
    save_img=True,
    line_thickness=2,
    region_thickness=4,
):
    vid_frame_count = 0

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # Setup Model
    model = YOLO(f"{weights}")
    model.to("cuda") if device == "0" else model.to("cpu")

    # Extract classes names
    names = model.model.names

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    # Output setup
    video_writer = cv2.VideoWriter(output, fourcc, fps, (frame_width, frame_height))
    count = 0
    # Iterate over video frames
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break

        # Extract the results
        results = model.track(frame, persist=True, show=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            clss = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Annotator Init and region drawing
            annotator = Annotator(frame, line_width=line_thickness)

            # Extract tracks
            for box, track_id, cls in zip(boxes, track_ids, clss):
                # Draw bounding box
                annotator.box_label(box, color=colors(int(cls), True))

                # Draw Tracks
                track_line = track_history[track_id]
                track_line.append(
                    (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
                )
                if len(track_line) > 30:
                    track_line.pop(0)

                prev_position = (
                    track_history[track_id][-2]
                    if len(track_history[track_id]) > 1
                    else None
                )

                # region: kanan atas OUT
                if (
                    prev_position is not None
                    and regionOut1["out1"].contains(Point(track_line[-1]))
                    and track_id not in out1_list
                ):
                    out1_list.append(track_id)
                    regionOut1["counts"] += 1
                # endregion

                # region: kanan atas OUT
                if (
                    prev_position is not None
                    and regionOut2["out1"].contains(Point(track_line[-1]))
                    and track_id not in out2_list
                ):
                    out2_list.append(track_id)
                    regionOut2["counts"] += 1
                # endregion

                # region: kanan atas OUT
                if (
                    prev_position is not None
                    and regionOut3["out1"].contains(Point(track_line[-1]))
                    and track_id not in out3_list
                ):
                    out3_list.append(track_id)
                    regionOut3["counts"] += 1
                # endregion

                # region: kanan atas OUT
                if (
                    prev_position is not None
                    and regionOut4["out1"].contains(Point(track_line[-1]))
                    and track_id not in out4_list
                ):
                    out4_list.append(track_id)
                    regionOut4["counts"] += 1
                # endregion

                # region: kiri atas IN
                if (
                    prev_position is not None
                    and regionIn1["in1"].contains(Point(track_line[-1]))
                    and track_id not in in1_list
                ):
                    in1_list.append(track_id)

                    if track_id in out1_list:
                        regionIn1["counts1"] += 1
                    if track_id in out2_list:
                        regionIn1["counts2"] += 1
                    if track_id in out3_list:
                        regionIn1["counts3"] += 1
                    if track_id in out4_list:
                        regionIn1["counts4"] += 1
                # endregion

                # region: atas kanan IN
                if (
                    prev_position is not None
                    and regionIn2["in1"].contains(Point(track_line[-1]))
                    and track_id not in in2_list
                ):
                    in2_list.append(track_id)

                    if track_id in out1_list:
                        regionIn2["counts1"] += 1
                    if track_id in out2_list:
                        regionIn2["counts2"] += 1
                    if track_id in out3_list:
                        regionIn2["counts3"] += 1
                    if track_id in out4_list:
                        regionIn2["counts4"] += 1
                # endregion

                # region: kanan bawah IN
                if (
                    prev_position is not None
                    and regionIn3["in1"].contains(Point(track_line[-1]))
                    and track_id not in in4_list
                ):
                    in4_list.append(track_id)

                    if track_id in out1_list:
                        regionIn3["counts1"] += 1
                    if track_id in out2_list:
                        regionIn3["counts2"] += 1
                    if track_id in out3_list:
                        regionIn3["counts3"] += 1
                    if track_id in out4_list:
                        regionIn3["counts4"] += 1
                # endregion

                # region: bawah kiri IN
                if (
                    prev_position is not None
                    and regionIn4["in1"].contains(Point(track_line[-1]))
                    and track_id not in in4_list
                ):
                    in4_list.append(track_id)

                    if track_id in out1_list:
                        regionIn4["counts1"] += 1
                    if track_id in out2_list:
                        regionIn4["counts2"] += 1
                    if track_id in out3_list:
                        regionIn4["counts3"] += 1
                    if track_id in out4_list:
                        regionIn4["counts4"] += 1
                # endregion

        regionOut1 = out_region[0]
        regionOut2 = out_region[1]
        regionOut3 = out_region[2]
        regionOut4 = out_region[3]
        regionIn1 = in_region[0]
        regionIn2 = in_region[1]
        regionIn3 = in_region[2]
        regionIn4 = in_region[3]

        # region: kanan atas OUT
        region_label1 = str(regionOut1["counts"])
        region_color = regionOut1["region_color"]
        region_text_color = regionOut1["text_color"]

        polygon_coords = np.array(regionOut1["out1"].exterior.coords, dtype=np.int32)
        centroid_x, centroid_y = int(regionOut1["out1"].centroid.x), int(
            regionOut1["out1"].centroid.y
        )

        text_size, _ = cv2.getTextSize(
            region_label1,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            thickness=line_thickness,
        )
        text_x = centroid_x - text_size[0] // 2
        text_y = centroid_y + text_size[1] // 2
        cv2.rectangle(
            frame,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            region_color,
            -1,
        )
        cv2.putText(
            frame,
            region_label1,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            region_text_color,
            line_thickness,
        )
        cv2.polylines(
            frame,
            [polygon_coords],
            isClosed=True,
            color=region_color,
            thickness=region_thickness,
        )
        # endregion

        # region: kanan atas OUT
        region_label2 = str(regionOut2["counts"])
        region_color = regionOut2["region_color"]
        region_text_color = regionOut2["text_color"]

        polygon_coords = np.array(regionOut2["out1"].exterior.coords, dtype=np.int32)
        centroid_x, centroid_y = int(regionOut2["out1"].centroid.x), int(
            regionOut2["out1"].centroid.y
        )

        text_size, _ = cv2.getTextSize(
            region_label2,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            thickness=line_thickness,
        )
        text_x = centroid_x - text_size[0] // 2
        text_y = centroid_y + text_size[1] // 2
        cv2.rectangle(
            frame,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            region_color,
            -1,
        )
        cv2.putText(
            frame,
            region_label2,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            region_text_color,
            line_thickness,
        )
        cv2.polylines(
            frame,
            [polygon_coords],
            isClosed=True,
            color=region_color,
            thickness=region_thickness,
        )
        # endregion

        # region: kiri bawah OUT
        region_label3 = str(regionOut3["counts"])
        region_color = regionOut3["region_color"]
        region_text_color = regionOut3["text_color"]

        polygon_coords = np.array(regionOut3["out1"].exterior.coords, dtype=np.int32)
        centroid_x, centroid_y = int(regionOut3["out1"].centroid.x), int(
            regionOut3["out1"].centroid.y
        )

        text_size, _ = cv2.getTextSize(
            region_label3,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            thickness=line_thickness,
        )
        text_x = centroid_x - text_size[0] // 2
        text_y = centroid_y + text_size[1] // 2
        cv2.rectangle(
            frame,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            region_color,
            -1,
        )
        cv2.putText(
            frame,
            region_label3,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            region_text_color,
            line_thickness,
        )
        cv2.polylines(
            frame,
            [polygon_coords],
            isClosed=True,
            color=region_color,
            thickness=region_thickness,
        )
        # endregion

        # region: atas kiri OUT
        region_label4 = str(regionOut4["counts"])
        region_color = regionOut4["region_color"]
        region_text_color = regionOut4["text_color"]

        polygon_coords = np.array(regionOut4["out1"].exterior.coords, dtype=np.int32)
        centroid_x, centroid_y = int(regionOut4["out1"].centroid.x), int(
            regionOut4["out1"].centroid.y
        )

        text_size, _ = cv2.getTextSize(
            region_label4,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            thickness=line_thickness,
        )
        text_x = centroid_x - text_size[0] // 2
        text_y = centroid_y + text_size[1] // 2
        cv2.rectangle(
            frame,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            region_color,
            -1,
        )
        cv2.putText(
            frame,
            region_label4,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            region_text_color,
            line_thickness,
        )
        cv2.polylines(
            frame,
            [polygon_coords],
            isClosed=True,
            color=region_color,
            thickness=region_thickness,
        )
        # endregion

        # region: kiri atas IN
        region_label10 = str(regionIn1["counts1"])
        region_label11 = str(regionIn1["counts2"])
        region_label12 = str(regionIn1["counts3"])
        region_label13 = str(regionIn1["counts4"])
        region_labels1 = [
            region_label10,
            region_label11,
            region_label12,
            region_label13,
        ]
        region_color = regionIn1["region_color"]

        polygon_coords = np.array(regionIn1["in1"].exterior.coords, dtype=np.int32)
        centroid_x, centroid_y = int(regionIn1["in1"].centroid.x), int(
            regionIn1["in1"].centroid.y
        )

        max_text_size = None

        for region_label in region_labels1:
            text_size, _ = cv2.getTextSize(
                region_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                thickness=line_thickness,
            )

            if max_text_size is None or text_size[0] > max_text_size[0]:
                max_text_size = text_size

        text_x = centroid_x - text_size[0] // 2
        text_y = centroid_y + text_size[1] // 2
        cv2.rectangle(
            frame,
            (text_x - 5, text_y - text_size[1] + 70),
            (text_x + max_text_size[0] + 5, text_y - 60),
            region_color,
            -1,
        )
        cv2.putText(
            frame,
            region_label10,
            (text_x, text_y - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            regionOut1["region_color"],
            line_thickness,
        )
        cv2.putText(
            frame,
            region_label11,
            (text_x, text_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            regionOut2["region_color"],
            line_thickness,
        )
        cv2.putText(
            frame,
            region_label12,
            (text_x, text_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            regionOut3["region_color"],
            line_thickness,
        )
        cv2.putText(
            frame,
            region_label13,
            (text_x, text_y + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            regionOut4["region_color"],
            line_thickness,
        )
        cv2.polylines(
            frame,
            [polygon_coords],
            isClosed=True,
            color=region_color,
            thickness=region_thickness,
        )
        # endregion

        # region: atas kanan IN
        region_label20 = str(regionIn2["counts1"])
        region_label21 = str(regionIn2["counts2"])
        region_label22 = str(regionIn2["counts3"])
        region_label23 = str(regionIn2["counts4"])
        region_labels2 = [
            region_label20,
            region_label21,
            region_label22,
            region_label23,
        ]
        region_color = regionIn2["region_color"]

        polygon_coords = np.array(regionIn2["in1"].exterior.coords, dtype=np.int32)
        centroid_x, centroid_y = int(regionIn2["in1"].centroid.x), int(
            regionIn2["in1"].centroid.y
        )

        max_text_size = None

        for region_label in region_labels2:
            text_size, _ = cv2.getTextSize(
                region_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                thickness=line_thickness,
            )

            if max_text_size is None or text_size[0] > max_text_size[0]:
                max_text_size = text_size

        text_x = centroid_x - text_size[0] // 2
        text_y = centroid_y + text_size[1] // 2
        cv2.rectangle(
            frame,
            (text_x - 5, text_y - text_size[1] + 80),
            (text_x + max_text_size[0] + 5, text_y - 60),
            region_color,
            -1,
        )
        cv2.putText(
            frame,
            region_label20,
            (text_x, text_y - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            regionOut1["region_color"],
            line_thickness,
        )
        cv2.putText(
            frame,
            region_label21,
            (text_x, text_y - 0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            regionOut2["region_color"],
            line_thickness,
        )
        cv2.putText(
            frame,
            region_label22,
            (text_x, text_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            regionOut3["region_color"],
            line_thickness,
        )
        cv2.putText(
            frame,
            region_label23,
            (text_x, text_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            regionOut4["region_color"],
            line_thickness,
        )
        cv2.polylines(
            frame,
            [polygon_coords],
            isClosed=True,
            color=region_color,
            thickness=region_thickness,
        )
        # endregion

        # region: kanan bawah IN
        region_label30 = str(regionIn3["counts1"])
        region_label31 = str(regionIn3["counts2"])
        region_label32 = str(regionIn3["counts3"])
        region_label33 = str(regionIn3["counts4"])
        region_labels2 = [
            region_label30,
            region_label31,
            region_label32,
            region_label33,
        ]
        region_color = regionIn3["region_color"]

        polygon_coords = np.array(regionIn3["in1"].exterior.coords, dtype=np.int32)
        centroid_x, centroid_y = int(regionIn3["in1"].centroid.x), int(
            regionIn3["in1"].centroid.y
        )

        max_text_size = None

        for region_label in region_labels2:
            text_size, _ = cv2.getTextSize(
                region_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                thickness=line_thickness,
            )

            if max_text_size is None or text_size[0] > max_text_size[0]:
                max_text_size = text_size

        text_x = centroid_x - text_size[0] // 2
        text_y = centroid_y + text_size[1] // 2
        cv2.rectangle(
            frame,
            (text_x - 5, text_y - text_size[1] + 70),
            (text_x + max_text_size[0] + 5, text_y - 60),
            region_color,
            -1,
        )
        cv2.putText(
            frame,
            region_label30,
            (text_x, text_y - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            regionOut1["region_color"],
            line_thickness,
        )
        cv2.putText(
            frame,
            region_label31,
            (text_x, text_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            regionOut2["region_color"],
            line_thickness,
        )
        cv2.putText(
            frame,
            region_label32,
            (text_x, text_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            regionOut3["region_color"],
            line_thickness,
        )
        cv2.putText(
            frame,
            region_label33,
            (text_x, text_y + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            regionOut4["region_color"],
            line_thickness,
        )
        cv2.polylines(
            frame,
            [polygon_coords],
            isClosed=True,
            color=region_color,
            thickness=region_thickness,
        )
        # endregion

        # region: bawah kiri IN
        region_label40 = str(regionIn4["counts1"])
        region_label41 = str(regionIn4["counts2"])
        region_label42 = str(regionIn4["counts3"])
        region_label43 = str(regionIn4["counts4"])
        region_labels4 = [
            region_label40,
            region_label41,
            region_label42,
            region_label43,
        ]
        region_color = regionIn4["region_color"]

        polygon_coords = np.array(regionIn4["in1"].exterior.coords, dtype=np.int32)
        centroid_x, centroid_y = int(regionIn4["in1"].centroid.x), int(
            regionIn4["in1"].centroid.y
        )

        max_text_size = None

        for region_label in region_labels4:
            text_size, _ = cv2.getTextSize(
                region_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                thickness=line_thickness,
            )

            if max_text_size is None or text_size[0] > max_text_size[0]:
                max_text_size = text_size

        text_x = centroid_x - text_size[0] // 2
        text_y = centroid_y + text_size[1] // 2
        cv2.rectangle(
            frame,
            (text_x - 5, text_y - text_size[1] + 70),
            (text_x + max_text_size[0] + 5, text_y - 60),
            region_color,
            -1,
        )
        cv2.putText(
            frame,
            region_label40,
            (text_x, text_y - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            regionOut1["region_color"],
            line_thickness,
        )
        cv2.putText(
            frame,
            region_label41,
            (text_x, text_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            regionOut2["region_color"],
            line_thickness,
        )
        cv2.putText(
            frame,
            region_label42,
            (text_x, text_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            regionOut3["region_color"],
            line_thickness,
        )
        cv2.putText(
            frame,
            region_label43,
            (text_x, text_y + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            regionOut4["region_color"],
            line_thickness,
        )
        cv2.polylines(
            frame,
            [polygon_coords],
            isClosed=True,
            color=region_color,
            thickness=region_thickness,
        )
        # endregion

        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow("Ultralytics YOLOv8 Region Counter Movable")
            cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame)

        if save_img:
            video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    del vid_frame_count
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
