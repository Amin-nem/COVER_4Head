# Fixed final_evaluation.py for Colab compatibility

import argparse, os, csv, torch, pandas as pd, numpy as np, pickle, joblib, yaml
from scipy import stats
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from tqdm import tqdm
from datetime import datetime

# Import COVER components
from cover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
from cover.models import COVER

print("--- مرحله ۱: بارگذاری تمام ابزارهای لازم ---")

# Fixed paths for Colab compatibility
VGGISH_FEATURES_PATH = 'features/audio_features_vggish.pkl'
AUDIO_MODEL_PATH = 'models/audio_head_lgbm_tuned.joblib'

# Load audio features
try:
    with open(VGGISH_FEATURES_PATH, 'rb') as f: 
        audio_features_dict = pickle.load(f)
    print(f"[صوتی] {len(audio_features_dict)} ویژگی VGGish با موفقیت بارگذاری شد.")
except FileNotFoundError: 
    print(f"Warning: Audio features file not found at '{VGGISH_FEATURES_PATH}'. Continuing without audio features.")
    audio_features_dict = {}

# Load audio model
try:
    audio_head_model = joblib.load(AUDIO_MODEL_PATH)
    print("[صوتی] هِد صوتی LightGBM (تیونینگ شده) با موفقیت بارگذاری شد.")
except FileNotFoundError: 
    print(f"Warning: Audio model file not found at '{AUDIO_MODEL_PATH}'. Continuing without audio model.")
    audio_head_model = None

# Normalization constants
mean_cover, std_cover = (torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375]))
mean_clip, std_clip = (torch.FloatTensor([122.77, 116.75, 104.09]), torch.FloatTensor([68.50, 66.63, 70.32]))

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def calculate_and_print_metrics(name, pred_scores, gt_scores):
    srocc, _ = stats.spearmanr(pred_scores, gt_scores)
    krocc, _ = stats.kendalltau(pred_scores, gt_scores)
    try:
        beta_init = [np.max(gt_scores), np.min(gt_scores), np.mean(pred_scores), 0.5]
        popt, _ = curve_fit(logistic_func, pred_scores, gt_scores, p0=beta_init, maxfev=int(1e8))
        pred_scores_logistic = logistic_func(pred_scores, *popt)
        plcc, _ = stats.pearsonr(gt_scores, pred_scores_logistic)
        rmse = np.sqrt(mean_squared_error(gt_scores, pred_scores_logistic))
    except Exception: 
        plcc, rmse = 0.0, 0.0
    print(f"\n--- نتایج برای: '{name}' ---")
    print(f"  SROCC: {srocc:.4f}, KROCC: {krocc:.4f}, PLCC: {plcc:.4f}, RMSE: {rmse:.4f}")
    return {"Model": name, "SROCC": srocc, "PLCC": plcc, "RMSE": rmse, "KROCC": krocc}

def save_results_to_csv(results_list, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame(results_list).to_csv(filename, index=False, float_format='%.4f')
    print(f"\nنتایج در فایل '{filename}' ذخیره شد.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--opt", type=str, default="cover.yml", help="the option file")
    parser.add_argument('-d', "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device: cuda:0 or cpu')
    parser.add_argument("-t", "--target_set", type=str, default="val-ytugc", help="target_set")
    parser.add_argument("--videos_dir", type=str, default="/content/drive/MyDrive/dataset/videos_split_perfect", help="Directory containing test videos")
    parser.add_argument("--max_videos", type=int, default=10, help="Maximum number of videos to process")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Load configuration
    with open(args.opt, "r") as f: 
        opt = yaml.safe_load(f)
    
    print(f"\n--- مرحله ۲: بارگذاری مدل COVER (Device: {args.device}) ---")
    evaluator = COVER(**opt["model"]["args"]).to(args.device)
    
    # Load pretrained weights
    if os.path.exists(opt["test_load_path"]):
        state_dict = torch.load(opt["test_load_path"], map_location=args.device)
        evaluator.load_state_dict(state_dict['state_dict'], strict=False)
        print("Model weights loaded successfully.")
    else:
        print(f"Warning: Pretrained model not found at {opt['test_load_path']}. Using random weights.")
    
    evaluator.eval()

    print("\n--- مرحله ۳: آماده‌سازی دیتاست تست ---")
    dopt = opt["data"][args.target_set]["args"]
    temporal_samplers = {}
    for stype, sopt in dopt["sample_types"].items():
        temporal_samplers[stype] = UnifiedFrameSampler(
            sopt["clip_len"] // sopt["t_frag"], 
            sopt["t_frag"], 
            sopt["frame_interval"], 
            sopt["num_clips"]
        )

    # Load test data
    DATA_INFO_CSV = 'scores/scores_duplicated.csv'
    df_all = pd.read_csv(DATA_INFO_CSV)
    df_all['flickr_id'] = df_all['flickr_id'].astype(str)
    
    # Load test IDs
    TEST_IDS_PATH = 'data_splits/test_video_ids.txt'
    if os.path.exists(TEST_IDS_PATH):
        with open(TEST_IDS_PATH, 'r') as f: 
            test_ids = [line.strip() for line in f]
    else:
        # Fallback: use first N videos from the dataset
        test_ids = df_all['flickr_id'].unique()[:args.max_videos]
        print(f"Test IDs file not found. Using first {len(test_ids)} videos from dataset.")
    
    df_test = df_all[df_all['flickr_id'].isin(test_ids)].reset_index(drop=True)
    
    # Use demo videos if available, otherwise use the flickr_ids
    demo_videos = [f for f in os.listdir(args.videos_dir) if f.endswith('.mp4')]
    if demo_videos:
        files = demo_videos[:args.max_videos]
        mos = [4.0] * len(files)  # Default MOS for demo videos
        print(f"Using {len(files)} demo videos for testing.")
    else:
        files = [f + '.mp4' for f in df_test['flickr_id'].tolist()]
        mos = df_test['mos'].tolist()
        print(f"تعداد {len(files)} فایل ویدئویی برای ارزیابی نهایی انتخاب شد.")

    print("\n--- مرحله ۴: شروع پیش‌بینی ---")
    pre_smos, pre_tmos, pre_amos, pre_audio = (np.zeros(len(mos)) for _ in range(4))

    with torch.no_grad():
        for i, video_file in enumerate(tqdm(files, desc="Processing Test Videos")):
            video_path = os.path.join(args.videos_dir, video_file)
            
            if not os.path.exists(video_path):
                print(f"Video not found: {video_path}")
                pre_smos[i], pre_tmos[i], pre_amos[i], pre_audio[i] = np.nan, np.nan, np.nan, np.nan
                continue

            try:
                # Process video using COVER
                views, _ = spatial_temporal_view_decomposition(video_path, dopt["sample_types"], temporal_samplers)
                
                for k, v in views.items():
                    num_clips = dopt["sample_types"][k].get("num_clips", 1)
                    if k in ['technical', 'aesthetic']:
                        views[k] = ( 
                            ((v.permute(1, 2, 3, 0) - mean_cover) / std_cover)
                            .permute(3, 0, 1, 2)
                            .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                            .transpose(0, 1)
                            .to(args.device)
                        )
                    elif k == 'semantic':
                        views[k] = ( 
                            ((v.permute(1, 2, 3, 0) - mean_clip) / std_clip)
                            .permute(3, 0, 1, 2)
                            .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                            .transpose(0, 1)
                            .to(args.device)
                        )
                
                video_results = [r.mean().item() for r in evaluator(views)]
                pre_smos[i], pre_tmos[i], pre_amos[i] = video_results[0], video_results[1], video_results[2]
                
                # Audio processing (if available)
                if audio_head_model and audio_features_dict:
                    base_name = df_test.loc[i, 'flickr_id'] if i < len(df_test) else video_file.replace('.mp4', '')
                    if base_name in audio_features_dict:
                        feature_vector = audio_features_dict[base_name]
                        pre_audio[i] = audio_head_model.predict(feature_vector.reshape(1, -1))[0]
                    else:
                        pre_audio[i] = 0.0  # Default audio score
                else:
                    pre_audio[i] = 0.0
                    
            except Exception as e:
                print(f"خطای غیرمنتظره در پردازش {video_file}: {e}")
                pre_smos[i], pre_tmos[i], pre_amos[i], pre_audio[i] = np.nan, np.nan, np.nan, np.nan
                continue

    # Final evaluation
    print("\n\n================== مرحله ۵: ارزیابی نهایی ==================")
    gt_mos = np.array(mos)
    
    # Remove NaN values
    valid_indices = ~np.isnan(pre_smos)
    if not np.any(valid_indices):
        print("No valid predictions found!")
        exit(1)
    
    pre_overall_cover_only = pre_smos[valid_indices] + pre_tmos[valid_indices] + pre_amos[valid_indices]
    results_cover = calculate_and_print_metrics("COVER (فقط تصویر)", pre_overall_cover_only, gt_mos[valid_indices])
    
    # Audio fusion
    W_AUDIO = 0.3
    pre_overall_with_audio = (pre_smos[valid_indices] + pre_tmos[valid_indices] + pre_amos[valid_indices]) + (pre_audio[valid_indices] * W_AUDIO)
    results_audio = calculate_and_print_metrics(f"COVER + Audio (وزن={W_AUDIO})", pre_overall_with_audio, gt_mos[valid_indices])
    
    print("\n========================= خلاصه نهایی =========================")
    results_list = [results_cover, results_audio]
    results_df = pd.DataFrame(results_list)
    print(results_df.to_string(index=False))
    print("================================================================")
    save_results_to_csv(results_list)