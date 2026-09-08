# Colab Keyframe Agent

Pipeline tìm keyframe theo phương pháp hai tầng có vòng lặp thích ứng:

```text
OpenAI lập query đầy đủ
        ↓
CLIP retrieval trực tiếp trên feature ZIP
        ↓
Chỉ trích JPG của shortlist, không unzip toàn bộ
        ↓
Ghép contact sheet để VLM so sánh nhiều candidate cùng lúc
        ↓
Nếu chưa đủ đúng: tạo feedback query yếu và retrieval lại
        ↓
Nếu có video mạnh nhưng ảnh nhỏ: Zoom Rescue, xem ít frame với ảnh lớn hơn
        ↓
Nếu vẫn không xác minh: truy vấn nới lỏng → chọn video → xem thưa
        ↓
Chọn mốc thời gian rồi chỉ quét keyframe cục bộ quanh các mốc đó
        ↓
Nếu cần chuyển động: chỉ trích đúng MP4, lấy mẫu đoạn ngắn
        ↓
Mở rộng keyframe trước/sau và VLM chọn frame cuối
        ↓
Map ordinal qua CSV → video_id + frame_idx
```

## Cấu trúc

```text
colab_keyframe_agent/
├── config.yaml                  # Mọi tham số có thể điều chỉnh
├── run.py                       # CLI
├── requirements.txt
├── notebooks/
│   └── colab_demo.ipynb
├── keyframe_agent/
│   ├── api/                     # OpenAI Responses API + prompts + schemas
│   ├── retrieval/               # CLIP encoder và feature-ZIP retrieval
│   ├── storage/                 # Mapping và giải nén chọn lọc
│   ├── visual/                  # Contact sheet và FFmpeg sampling
│   ├── config.py
│   ├── models.py
│   └── pipeline.py              # Vòng điều khiển thích ứng
└── tests/
```

## Dữ liệu được hỗ trợ

Cấu hình mặc định khớp bộ AIC hiện tại:

- `clip-features-32-aic25-b1.zip`
- `map-keyframes-aic25-b1.zip`
- `Keyframes_*.zip`
- `Videos_*.zip`

Feature ZIP được đọc trực tiếp. Chương trình chỉ trích các thành viên cần thiết từ
keyframe/video ZIP vào cache.

## Chạy trên Google Colab

### 1. Đưa thư mục lên Google Drive

Ví dụ:

```text
MyDrive/AICHALLENGENHCM2026/colab_keyframe_agent
```

### 2. Mount Drive và cài dependency

```python
from google.colab import drive
drive.mount('/content/drive')

%cd "/content/drive/MyDrive/AICHALLENGENHCM2026/colab_keyframe_agent"
!pip install -q -r requirements.txt
!apt-get update -qq && apt-get install -y -qq ffmpeg
```

### 3. Cấu hình API key

Nên lưu `OPENAI_API_KEY` trong Colab Secrets, sau đó:

```python
import os
from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
```

Không ghi API key trực tiếp vào notebook hoặc `config.yaml`.

### 4. Sửa đường dẫn

Mở `config.yaml`, đặt `data.root` tới thư mục chứa các ZIP. Mặc định:

```yaml
data:
  root: "/content/drive/MyDrive/AICHALLENGENHCM2026/AI CHALLENGEN HCM2026"
```

### 5. Chạy

```python
from keyframe_agent import KeyframeSearchPipeline, load_config

config = load_config("config.yaml")
pipeline = KeyframeSearchPipeline(config)
result = pipeline.run("Sân khấu với dòng chữ nổi 3D ánh kim...")
result
```

Hoặc CLI:

```bash
python run.py --config config.yaml --query "nội dung query"
```

## Chạy retrieval-only

Chế độ này không gọi OpenAI và chỉ tạo shortlist/contact sheet:

```bash
python run.py --config config.yaml --retrieval-only --query "English visual query"
```

Trong retrieval-only, query planner bị bỏ qua. Vì OpenAI CLIP thường hoạt động tốt hơn
với tiếng Anh, hãy nhập một câu tiếng Anh đầy đủ, không dùng các mảnh từ khóa rời.

## Đầu ra

Mỗi query tạo một thư mục riêng trong `data.output_dir`:

```text
20260830_120000_query/
├── query_plan.json
├── round_01/
│   ├── candidates.json
│   ├── retrieval_01.jpg
│   └── vlm_ranking.json
├── rescue_zoom/                # Có khi primary có video mạnh nhưng chưa xác minh
│   ├── zoom_full_01.jpg         # Ảnh full frame, ít card hơn và lớn hơn
│   ├── zoom_detail_01.jpg       # Crop từ chính các frame ở zoom_full
│   └── vlm_final_selection.json
├── fallback_video_scan/         # Chỉ có khi primary và Rescue Zoom chưa xác minh được
│   ├── relaxed_retrieval_candidates.json
│   ├── seed_videos.json
│   ├── representatives/
│   │   ├── video_representatives_01.jpg
│   │   └── vlm_video_selection.json
│   └── local_scan/<video_id>/batch_001/
│       ├── local_scan_01.jpg
│       └── vlm_ranking.json
├── motion/                      # Chỉ có khi query cần chuyển động
├── final_neighbors/
│   ├── neighbors_01.jpg
│   └── vlm_final_selection.json
├── result.json
└── result.csv
```

`result.json` phân biệt rõ:

```json
{
  "video_id": "L30_V026",
  "frame_idx": 731,
  "keyframe_ordinal": 38,
  "pts_time": 29.24,
  "confidence": 0.91,
  "verified": true,
  "keyframes": [
    {
      "video_id": "L30_V026",
      "frame_idx": 731,
      "keyframe_ordinal": 38,
      "confidence": 0.91,
      "verified": true
    }
  ]
}
```

`frame_idx` là frame nguồn được map qua CSV. `keyframe_ordinal` chỉ là thứ tự ảnh
trong bộ keyframe. `keyframes` chứa tối đa 5–10 phương án do VLM xếp hạng;
phương án đầu là đáp án chính. Mỗi phương án có `verified` riêng nên các frame gần
giống vẫn có thể được hiển thị mà không bị tuyên bố nhầm là đáp án chính xác.

## Xác minh điều kiện có kiểu

Query planner không hardcode nội dung như `SẮC CỔ` hoặc `Hồ Tùng Mậu`. Nó tự tách
mỗi query thành các criterion tổng quát:

- `visual`: người, vật, màu sắc, hành động và bố cục nhìn thấy được;
- `exact_text`: nội dung chữ hoặc số phải đọc chính xác;
- `text_presence`: phải thấy biển/chữ nhưng không cần chép chính xác;
- `count`: số lượng nhìn thấy là điều kiện quyết định;
- `temporal`: thứ tự hoặc chuyển động cần nhiều frame;
- `context`: tên địa điểm/sự kiện dùng tìm nguồn, không bắt buộc phải hiện thành chữ.

Code chỉ hard-gate criterion `required` có loại nằm trong
`verification.hard_gate_types` và planner đủ chắc về phân loại. Vì mặc định không
hard-gate `context`, câu “rẽ vào đường Hồ Tùng Mậu” không bị loại chỉ vì ảnh không
có biển tên đường. Ngược lại, câu yêu cầu dòng chữ “SẮC CỔ” sẽ được phân loại
`exact_text` và phải có bằng chứng đọc được.

## TRAKE nhiều event: E1, E2, E3, ...

Query TRAKE có nhiều event được chuyển sang nhánh `multi_event_trake`, không đi qua bộ chọn
một frame. Hệ thống mã hóa CLIP riêng cho từng event trong một lần quét feature,
xếp hạng video theo event yếu nhất, rồi dùng VLM kiểm tra các frame trong cùng video.

Kết quả chứa:

- `events`: một phần tử cho mỗi E1/E2/... theo đúng thứ tự;
- `events[].selected`: frame đáp án của event;
- `events[].alternatives`: 5–10 frame được xếp hạng riêng cho event;
- `chronological`: chỉ `true` khi thời gian các frame tăng nghiêm ngặt;
- `keyframes`: danh sách gọn các frame được chọn cho tất cả event.

Một ảnh vườn hoặc mâm nhiều loại trái cây không thể thay thế cho bốn cảnh sầu riêng,
măng cụt, bưởi và bòn bon. `verified=true` chỉ khi đủ mọi event, cùng một video và
đúng thứ tự thời gian.

Nếu các điểm CLIP đầu tiên chỉ tìm được một phần chuỗi, pipeline xếp video theo
`event_coverage` thay vì độ tự tin từ chối của VLM. Mặc định nó chỉ lấy video có độ
phủ cao nhất và bổ sung 48 keyframe rải đều toàn timeline. Nhờ vậy một event CLIP
nhận diện yếu vẫn có cơ hội xuất hiện mà không phải quét sâu tất cả video.

Nên chỉ định loại bài trực tiếp khi chạy:

```python
# KIS: tìm một keyframe phù hợp
result = pipeline.run(query, task_type="kis")

# TRAKE có chuỗi E1, E2, ...
result = pipeline.run(query, task_type="trake")

# QA: trả về 5 video có bằng chứng, câu trả lời chỉ được điền khi nhìn thấy được
result = pipeline.run(query, task_type="qa")

# QA nhiều cảnh cách xa nhau: ép thủ công nếu muốn bỏ qua nhận diện tự động
result = pipeline.run(
    query,
    task_type="qa",
    qa_mode="long_range_temporal",
)
```

`task_type="auto"` vẫn được hỗ trợ để tương thích: planner nhận diện QA khi câu truy
vấn có câu hỏi, và TRAKE khi có ít nhất hai event có yêu cầu thứ tự. Chỉ định trực tiếp
vẫn đáng tin cậy hơn.

Với QA, hãy luôn giữ cả mô tả cảnh lẫn câu hỏi trong `query`, ví dụ `... tổng cộng 4
con cá. Đây là loài cá gì?`. Luồng QA tìm riêng từng tiêu chí, xếp video theo độ phủ,
rồi lấy 24 keyframe rải đều cộng với các hit CLIP của từng tiêu chí. VLM trả đúng 5
gói bằng chứng mỗi round. Mỗi gói có `video_id`, các `frame_idx`
thật, `answer`, `answer_source`, `count_status` và `verified`. `count_status` phân
biệt bốn vật thể cùng một frame (`same_frame`) với tổng số được chứng minh qua nhiều
frame (`across_frames`). Nếu tên loài chỉ được nói trong âm thanh hoặc không nhìn thấy,
`answer` sẽ là `null`, không đoán tên.

Thứ tự ưu tiên xác định loại bài là:

1. `pipeline.run(query, task_type="...")`;
2. `runtime.task_type_override` trong `config.yaml`;
3. planner tự nhận diện khi cả hai giá trị trên là `auto`.

### QA thích ứng: `single_frame`, `short_window`, `long_range_temporal`

Mặc định `qa_mode="auto"`: planner LLM tự chọn cơ chế nội bộ, còn
`task_type="qa"` vẫn giữ nguyên. Có thể ép mode ở từng câu bằng tham số
`pipeline.run(..., qa_mode="...")`, hoặc đặt `runtime.qa_mode_override` trong
`config.yaml`. Thứ tự ưu tiên là tham số của `run` → config → planner.

- `single_frame`: cảnh và đáp án thường cùng nằm trong một ảnh.
- `short_window`: cần vài frame gần nhau để xác minh hành động hoặc tổng số.
- `long_range_temporal`: các cảnh nhận diện video nằm xa nhau, hoặc frame chứa đáp
  án nằm ở đoạn khác. Nhánh này giữ lại top video theo toàn query, thêm truy vấn cho
  từng event/answer locator và lấy 48 frame rải đều trên timeline.

`long_range_temporal` chỉ chạy khi được nhận diện hoặc ép rõ ràng, nên không làm tăng
chi phí cho các QA thông thường. Kết quả có thêm `qa_mode`, và từng bundle có
`answer_evidence_frames` để phân biệt frame chứa đáp án với frame xác nhận câu chuyện.
Trường `qa_mode_source` cho biết mode đến từ `planner`, `run_argument`,
`config_override` hay bộ heuristic dự phòng.

## KIS thích ứng: static và temporal

`task_type="kis"` vẫn là một loại đầu ra: một keyframe chính và 5–10 phương án.
Pipeline tự chọn cơ chế nội bộ dựa trên query plan:

- `static_kis`: mọi criterion bắt buộc thuộc `frame`; chạy nguyên luồng
  primary → Rescue Zoom → fallback cũ.
- `temporal_kis`: có `needs_motion=true`, criterion `temporal`, hoặc
  `evidence_scope` là `window/video`; dùng nhiều frame để xác minh video rồi chọn
  một frame đại diện.

Temporal KIS tìm riêng từng criterion, screening video, và tối đa hai round xác minh.
Tất cả bằng chứng bắt buộc phải thuộc cùng một video. Criterion temporal còn phải có
`pts_time` tăng đúng thứ tự. Frame cuối không phải chứa cả câu chuyện, nhưng phải là
một frame đã được trích dẫn cho khoảnh khắc quan trọng. Kết quả có thêm
`kis_mode`, `video_verified`, `frame_verified`, `evidence_frames` và
`temporal_kis_attempts`.

## Tổng hợp tham số trong `config.yaml`

### Đường dẫn và chạy cơ bản

| Nhóm | Tham số | Ý nghĩa |
|---|---|---|
| `data` | `root`, `feature_zip`, `mapping_zip` | Thư mục dữ liệu và hai ZIP bắt buộc. |
| `data` | `keyframe_zip_glob`, `video_zip_glob` | Mẫu tên ZIP keyframe/video. Chỉ đổi khi bộ dữ liệu đổi tên. |
| `data` | `cache_dir`, `output_dir` | Cache giải nén chọn lọc và nơi ghi kết quả. |
| `runtime` | `retrieval_only` | `true` để chỉ tạo shortlist, không gọi OpenAI. |
| `runtime` | `keep_extracted_videos` | Giữ MP4 đã trích khi debug chuyển động; tốn dung lượng hơn. |

### Retrieval và contact sheet đầu tiên

| Tham số | Tăng để | Giảm để |
|---|---|---|
| `retrieval.main_weight` | bám chặt query đầy đủ hơn | cho support query ảnh hưởng nhiều hơn |
| `retrieval.support_weight`, `max_support_queries` | dùng nhiều gợi ý phụ/feedback hơn | tránh query phụ làm lệch kết quả |
| `retrieval.top_per_video` | lấy thêm cảnh từ một video | đa dạng video hơn |
| `retrieval.shortlist_size` | tăng recall của vòng đầu | giảm số ảnh/API |
| `retrieval.max_candidates_per_video` | cho một video chiếm nhiều slot hơn | chống một video áp đảo shortlist |
| `retrieval.max_rounds` | thêm lượt retrieval → VLM → feedback | giảm thời gian và chi phí |
| `retrieval.visual_accept_score` | dừng sớm chỉ khi VLM chắc hơn | dừng sớm dễ hơn |
| `contact_sheet.columns`, `cards_per_sheet` | nhiều card/sheet, ít ảnh gửi API hơn | ít card/sheet, từng ảnh dễ quan sát hơn |
| `contact_sheet.card_width`, `card_height`, `image_height` | chữ/vật nhỏ rõ hơn | giảm byte ảnh và chi phí vision |
| `contact_sheet.jpeg_quality` | giữ chi tiết JPEG tốt hơn | giảm dung lượng ảnh |

`clip_model` và `device` thường giữ nguyên; chỉ đổi `device` khi cần ép chạy CPU/GPU.

### OpenAI và chuyển động

| Nhóm | Tham số | Cách chỉnh |
|---|---|---|
| `openai` | `api_key_env` | Tên biến môi trường chứa API key; không ghi key vào YAML. |
| `openai` | `model` | Đổi model khi cần thay trade-off chất lượng/chi phí. |
| `openai` | `reasoning_effort`, `image_detail` | Tăng cho query khó/chữ nhỏ; tăng chi phí và độ trễ. |
| `openai` | `max_output_tokens`, `retries`, `retry_base_seconds` | Tăng token khi JSON/phân tích dài; tăng retry khi mạng không ổn định. |
| `openai` | `final_candidate_count` | Chọn từ 5–10 frame xếp hạng ở kết quả cuối; tăng để có nhiều phương án so sánh, giảm để JSON gọn hơn. |
| `refinement` | `finalists`, `neighbor_radius` | Tăng để xem nhiều candidate/frame kề nhau hơn. |
| `refinement` | `use_raw_video_for_motion`, `raw_video_finalists` | Chỉ bật/tăng khi query thật sự cần diễn biến theo thời gian. |
| `refinement` | `video_window_before`, `video_window_after`, `static_fps`, `motion_fps`, `max_sampled_frames` | Tăng để bắt hành động ngắn tốt hơn; tốn trích MP4 và API hơn. |
| `refinement` | `final_confidence_threshold` | Tăng để nghiêm ngặt hơn; giảm để nhận nhiều kết quả hơn. |
| `verification` | `hard_gate_types` | Các loại criterion bắt buộc phải vượt kiểm chứng; thường không thêm `context`. |
| `verification` | `classification_confidence_threshold` | Tăng để chỉ hard-gate khi planner rất chắc về cách phân loại; giảm để áp gate rộng hơn. |
| `verification` | `criterion_confidence_threshold` | Tăng để đòi bằng chứng từng điều kiện chắc hơn; giảm để bớt nghiêm. |
| `multi_event` | `candidate_videos` | Tăng để VLM thử thêm video chứa đủ chuỗi; tăng chi phí API gần tuyến tính. |
| `multi_event` | `top_frames_per_event_per_video` | Tăng để giữ thêm điểm CLIP cho từng E; hữu ích khi một loại trái xuất hiện nhiều lần. |
| `multi_event` | `neighbor_radius` | Tăng để bắt cảnh đầu tiên/chuyển cảnh quanh điểm CLIP; đồng thời gửi nhiều ảnh hơn. |
| `multi_event` | `max_candidates_per_video` | Giới hạn tổng ảnh của một video được gửi để xác minh chuỗi. |
| `multi_event` | `uniform_fallback_enabled` | Bật một lượt quét đều khi candidate CLIP chưa đủ event. |
| `multi_event` | `uniform_fallback_videos` | Số video có độ phủ cao nhất được cứu; tăng sẽ tăng lượt API. |
| `multi_event` | `uniform_fallback_frames` | Số frame rải đều trên toàn video; tăng để bắt cảnh ngắn nhưng gửi nhiều ảnh hơn. |
| `multi_event` | `uniform_fallback_max_candidates` | Trần tổng candidate trong lượt cứu TRAKE. |

OpenAI models trong pipeline nhận ảnh, không nhận MP4 trực tiếp. MP4 được lấy mẫu thành
ảnh có timestamp trước khi gửi API.

### QA — 5 gói bằng chứng theo video

Nhánh này dùng cho câu có đáp án cần tìm, không phải chỉ tìm một ảnh. CLIP chạy một
lần trên toàn bộ feature nhưng chấm riêng query đầy đủ và từng criterion thị giác.
Video được xếp theo số criterion được phủ, chất lượng bằng chứng và điểm query đầy đủ.
Mỗi round VLM xem đúng 5 video. Nếu round đầu không có kết quả vượt hard gate, round
hai tự động thử 5 video kế tiếp. `result_count` và `candidate_videos` giữ bằng 5 để
mọi video gửi vào đều được VLM đánh giá.

| Tham số | Tăng để | Giảm để |
|---|---|---|
| `qa.retrieval_shortlist_size` | giữ bảng xếp hạng chẩn đoán dài hơn | JSON chẩn đoán gọn hơn; không được thấp hơn `candidate_videos * max_rounds` |
| `qa.candidate_videos` | giữ bằng `result_count=5` để mỗi video đều được đánh giá | không nên chỉnh riêng lẻ |
| `qa.max_rounds` | thử thêm nhóm 5 video khi chưa verified | giảm API; có thể bỏ lỡ video xếp sau |
| `qa.fallback_on_unverified` | bật round tiếp theo khi tất cả đều trượt hard gate | tắt để luôn chỉ gọi một round |
| `qa.screening_enabled` | bật VLM lọc thưa trước khi QA chi tiết | tắt để dùng thẳng thứ hạng CLIP |
| `qa.screening_videos` | cho bước lọc thưa xem sâu hơn trong bảng CLIP | giảm số video và số contact sheet screening |
| `qa.screening_frames_per_video` | cho screening thấy thêm loại bằng chứng trong mỗi video | giảm chi phí ảnh ở bước screening |
| `qa.max_retrieval_criteria` | dùng thêm điều kiện độc lập để tính độ phủ | giảm thời gian encode và số hit giữ lại |
| `qa.top_frames_per_criterion_per_video` | giữ thêm frame mạnh cho mỗi criterion | giảm số ảnh candidate |
| `qa.coverage_rank_cutoff` | coi nhiều video hơn là có phủ một criterion | yêu cầu từng criterion xếp hạng chặt hơn |
| `qa.coverage_weight` | ưu tiên video có nhiều dấu hiệu khác nhau | giảm ảnh hưởng độ phủ |
| `qa.evidence_score_weight` | ưu tiên chất lượng tổng hợp của các dấu hiệu riêng | giảm ảnh hưởng điểm criterion riêng |
| `qa.main_query_weight` | bám sát toàn bộ câu gốc hơn | ưu tiên bằng chứng phân tán hơn |
| `qa.representatives_per_video` | bao phủ timeline tốt hơn, hữu ích cho thao tác/đếm theo đoạn | ít frame gửi VLM hơn |
| `qa.max_candidates_per_video` | giữ thêm hit CLIP ngoài các frame rải đều | giới hạn ảnh một video chiếm slot |
| `qa.minimum_confidence` | chỉ gắn `verified=true` khi bằng chứng chắc hơn | dễ đánh dấu candidate là verified hơn |

`evidence_scope` được planner gán riêng từng tiêu chí: `frame` (một ảnh), `window`
(vài frame lân cận) hoặc `video` (tổng hợp cả đoạn). Vì vậy “4 con cá” không còn bị
buộc phải nhìn thấy đồng thời trong duy nhất một keyframe. Tuy nhiên tên loài cá chỉ
được điền khi chữ hoặc dấu hiệu thị giác đủ rõ; luồng này không suy luận từ âm thanh.

### Temporal KIS — chỉ dùng cho KIS kể diễn biến

| Tham số | Tăng để | Giảm để |
|---|---|---|
| `temporal_kis.max_rounds` | thử thêm nhóm video khi round đầu thất bại | giảm số lượt VLM |
| `temporal_kis.screening_videos` | tăng recall trước bước kiểm tra chi tiết | giảm ảnh screening |
| `temporal_kis.screening_frames_per_video` | thấy thêm khoảnh khắc của mỗi video | giảm image tokens |
| `temporal_kis.max_retrieval_criteria` | dùng thêm chi tiết độc lập để xếp độ phủ | retrieval gọn hơn |
| `temporal_kis.top_frames_per_criterion_per_video` | giữ thêm mốc CLIP cho mỗi hành động | giảm candidate |
| `temporal_kis.coverage_rank_cutoff` | xem nhiều video là có phủ criterion | bắt criterion chặt hơn |
| `temporal_kis.coverage_weight` | ưu tiên video có nhiều chi tiết khác nhau | tăng tương đối ảnh hưởng điểm CLIP |
| `temporal_kis.evidence_score_weight` | ưu tiên chất lượng các hit criterion | giảm ảnh hưởng hit riêng |
| `temporal_kis.main_query_weight` | bám mạnh hơn vào toàn bộ câu gốc | ưu tiên chi tiết phân tán hơn |
| `temporal_kis.representatives_per_video` | bao phủ timeline dày hơn | giảm ảnh gửi VLM |
| `temporal_kis.max_candidates_per_video` | giữ thêm frame rải đều/hit criterion | giới hạn chi phí từng video |
| `temporal_kis.minimum_confidence` | khó gắn `verified=true` hơn | chấp nhận kết quả dễ hơn |

Ba trọng số coverage/evidence/main phải có tổng bằng `1.0`. Đặt
`temporal_kis.enabled: false` để mọi KIS quay lại hoàn toàn cơ chế cũ.

### Rescue Zoom — nên giữ bật

Rescue Zoom chỉ chạy khi primary đã có một video mạnh nhưng chưa xác minh được. Nó giữ
vài cụm frame mạnh, làm ảnh lớn hơn và gửi một lượt VLM xác minh riêng. Crop chỉ là
phóng to từ **chính candidate đó**, không được dùng để ghép hai frame khác nhau.

| Tham số | Tăng để | Giảm để |
|---|---|---|
| `enabled` | — | tắt hẳn lượt cứu chi phí thấp này |
| `min_primary_score` | chỉ zoom khi primary chắc hơn | zoom nhiều case hơn |
| `preserve_primary_videos` | giữ thêm video primary mạnh | giới hạn scope rescue |
| `max_candidates`, `neighbor_radius` | xem thêm frame/cụm lân cận | giới hạn ảnh rescue |
| `columns`, `cards_per_sheet` | nhiều ảnh/sheet hơn | ảnh lớn hơn, đọc chữ tốt hơn |
| `card_width`, `card_height`, `image_height` | chi tiết/chữ rõ hơn | rẻ hơn |
| `crop_text_when_needed` | bật/tắt crop khi query có chữ | giảm thêm ảnh gửi API |
| `text_detail_regions` | thêm `top`, `center`, `bottom` khi chữ có thể ở vùng đó | chỉ giữ `bottom` để rẻ nhất |

Mặc định `min_primary_score: 75`, tối đa 16 candidate và 8 card/sheet là điểm cân bằng
tốt. Với query kiểu sân khấu/chữ ở mép dưới, giữ `text_detail_regions: ["bottom"]`.

### Fallback quét cục bộ — lớp cuối

Fallback chỉ chạy sau primary **và** Rescue Zoom. Nó không quét toàn bộ video mặc định:
truy vấn nới lỏng tìm video, xem thưa để chọn mốc, sau đó chỉ xem cửa sổ keyframe quanh
mốc đó.

| Tham số | Tăng để | Giảm để |
|---|---|---|
| `enabled`, `trigger_on_unverified` | bật fallback khi primary chưa chắc | chỉ dùng primary/rescue |
| `relaxed_query_enabled`, `relaxed_shortlist_size` | recall rộng hơn, ít bỏ sót video đúng | fallback nhanh hơn |
| `candidate_videos`, `representative_frames_per_video` | xem nhiều video/điểm thời gian hơn | giảm chi phí chọn video |
| `selected_videos_for_deep_scan` | xử lý thêm video (tên cũ, hiện là số video quét cục bộ) | giảm số lượt VLM |
| `preserve_primary_videos` | không để fallback bỏ quên video mạnh từ primary | scope fallback nhỏ hơn |
| `local_scan_anchors_per_video`, `local_scan_radius` | quét thêm mốc và frame quanh mốc | giảm frame gửi API |
| `deep_scan_batch_size` | ít API call hơn nhưng ảnh/batch nhiều hơn | batch nhỏ, dễ nhìn hơn |
| `deep_scan_keep_per_video` | giữ nhiều tâm để xác minh cuối | kết quả cuối gọn/rẻ hơn |

Preset tiết kiệm có thể dùng:

```yaml
rescue_zoom:
  max_candidates: 12
  cards_per_sheet: 6

fallback:
  candidate_videos: 8
  representative_frames_per_video: 12
  selected_videos_for_deep_scan: 1
  local_scan_anchors_per_video: 1
  local_scan_radius: 10
```

## Nguyên tắc giữ độ chính xác

1. Query chính luôn chứa toàn bộ bố cục và có trọng số cao nhất.
2. Support/feedback query chỉ hỗ trợ, không được thay query chính.
3. VLM so sánh ứng viên cùng lúc thay vì xác minh độc lập từng frame.
4. Nếu video primary mạnh nhưng ảnh nhỏ, Rescue Zoom kiểm tra lại ít frame bằng ảnh
   lớn hơn trước khi mở rộng phạm vi tìm kiếm.
5. Frame gần đúng được mở rộng trước/sau; crop chỉ được xem như bằng chứng của chính
   frame đó.
6. Video gốc chỉ được trích cho finalist cần chuyển động.
7. Khi shortlist sai video, fallback xem thưa rồi quét cục bộ quanh mốc thời gian;
   nó không lấy frame top-1 của CLIP làm đáp án.
8. Kết quả cuối luôn map qua `map-keyframes` trước khi trả `frame_idx`.

## Chi phí và riêng tư

Contact sheet và các frame refinement được gửi tới OpenAI API. Không có toàn bộ ZIP
hay toàn bộ video được tải lên. Hãy điều chỉnh `shortlist_size`, kích thước sheet và
`max_sampled_frames` để kiểm soát chi phí, đồng thời bảo đảm việc gửi hình phù hợp
với quyền sử dụng dữ liệu của bạn.

Phần gọi API sử dụng OpenAI Responses API với image inputs và Structured Outputs,
theo [tài liệu OpenAI chính thức](https://developers.openai.com/api/reference/cli/resources/responses/methods/create).
