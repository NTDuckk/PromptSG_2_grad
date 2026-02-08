import logging
import os
import time
import torch
import torch.nn as nn
from torch.cuda import amp
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
import torch.distributed as dist
from torch.nn import functional as F
import subprocess
import sys
import csv

def setup_training_logger(cfg):
    """Setup additional file logger for training metrics"""
    # Create logs directory if not exists
    log_dir = cfg.OUTPUT_DIR
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup file logger for metrics
    metrics_logger = logging.getLogger("promptsg.metrics")
    metrics_logger.setLevel(logging.INFO)
    
    # File handler for metrics
    metrics_file = os.path.join(log_dir, 'training_metrics.txt')
    file_handler = logging.FileHandler(metrics_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    if not metrics_logger.handlers:
        metrics_logger.addHandler(file_handler)
    
    return metrics_logger

def auto_generate_plots(cfg):
    """Automatically generate learning curves after training completion"""
    logger = logging.getLogger("promptsg.train")
    logger.info("Generating learning curves...")
    
    try:
        # Get the script directory and plot script path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        plot_script = os.path.join(project_root, 'plot_learning_curves.py')
        
        if not os.path.exists(plot_script):
            logger.warning(f"Plot script not found at {plot_script}")
            return False
        
        # Prepare command
        cmd = [
            sys.executable,  # Use current Python interpreter
            plot_script,
            '--log_dir', cfg.OUTPUT_DIR,
            '--output_dir', os.path.join(project_root, 'plots'),
            '--save_json'
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run the plot script
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode == 0:
            logger.info("Learning curves generated successfully!")
            logger.info(f"Output: {result.stdout}")
            if result.stderr:
                logger.info(f"Warnings: {result.stderr}")
            return True
        else:
            logger.error(f"Failed to generate learning curves: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Plot generation timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"Error generating plots: {str(e)}")
        return False

def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn, num_query, local_rank):
    log_period = cfg.SOLVER.PROMPTSG.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.PROMPTSG.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.PROMPTSG.EVAL_PERIOD
    epochs = cfg.SOLVER.PROMPTSG.MAX_EPOCHS

    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("promptsg.train")
    logger.info('start training')
    logger.info("Config:\n{}".format(cfg.dump()))

    # Setup metrics file logger
    metrics_logger = setup_training_logger(cfg)
    metrics_logger.info("=== TRAINING STARTED ===")
    metrics_logger.info(f"Model: {cfg.MODEL.NAME}")
    metrics_logger.info(f"Prompt mode: {cfg.MODEL.PROMPTSG.PROMPT_MODE}")
    metrics_logger.info(f"Max epochs: {epochs}")
    metrics_logger.info(f"Learning rate: {cfg.SOLVER.PROMPTSG.BASE_LR_VISUAL}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics_logger.info(f"Total parameters: {total_params:,}")
    metrics_logger.info(f"Trainable parameters: {trainable_params:,}")
    metrics_logger.info("="*50)

    if device:
        device = torch.device(f"cuda:{local_rank}") if local_rank is not None else torch.device("cuda")
        model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    id_meter = AverageMeter()
    tri_meter = AverageMeter()
    supcon_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    all_start_time = time.monotonic()

    # CSV file for structured metrics (used by plot_learning_curves.py)
    csv_path = os.path.join(cfg.OUTPUT_DIR, 'metrics.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'total_loss', 'id_loss', 'tri_loss', 'supcon_loss', 'acc', 'lr', 'mAP', 'rank1', 'rank5', 'rank10'])
    csv_file.flush()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset(); acc_meter.reset(); evaluator.reset()
        id_meter.reset(); tri_meter.reset(); supcon_meter.reset()

        logger.info("Epoch {} started".format(epoch))
        metrics_logger.info(f"EPOCH {epoch} - Training started")
        model.train()

        for n_iter, (img, pid, camid, viewid) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            target = pid.to(device)

            with amp.autocast(enabled=True):
                # cls_score, triplet_feats, image_feat, text_feat = model(img, target)
                cls_score, triplet_feats, image_feat, text_feat = model(x = img, label = target)

                # ============ DEBUG: Check model outputs ============
                if epoch == 1 and n_iter == 0:
                    logger.info("=== DEBUG: Model Outputs ===")
                    if isinstance(cls_score, (list, tuple)):
                        logger.info(f"Number of cls_scores: {len(cls_score)}")
                        for i, score in enumerate(cls_score):
                            logger.info(f"  cls_score[{i}] shape: {score.shape}")
                    else:
                        logger.info(f"cls_score shape: {cls_score.shape}")
                    logger.info(f"triplet_feats type: {type(triplet_feats)}")
                    if isinstance(triplet_feats, (list, tuple)):
                        logger.info(f"Number of triplet features: {len(triplet_feats)}")
                        for i, feat in enumerate(triplet_feats):
                            logger.info(f"  triplet_feats[{i}] shape: {feat.shape}")
                            logger.info(f"  triplet_feats[{i}] min/max: {feat.min().item():.4f}/{feat.max().item():.4f}")
                    else:
                        logger.info(f"triplet_feats shape: {triplet_feats.shape}")
                    logger.info(f"image_feat shape: {image_feat.shape}")
                    logger.info(f"text_feat shape: {text_feat.shape}")
                    logger.info(f"target shape: {target.shape}")
                    logger.info(f"Batch size: {img.shape[0]}")
                    logger.info("===========================")
                # ============ END DEBUG ============

                total_loss, losses_dict = loss_fn(cls_score, triplet_feats, target, camid, image_feat, text_feat)
                loss = total_loss
                id_loss = losses_dict['id_loss']
                tri_loss = losses_dict['tri_loss']
                supcon_loss = losses_dict['supcon_loss']
                
                # ============ DEBUG: Check loss values ============
                if epoch == 1 and n_iter == 0:
                    logger.info("=== DEBUG: Loss Values ===")
                    logger.info(f"Total loss: {loss.item():.6f}")
                    logger.info(f"ID loss: {id_loss.item():.6f}")
                    logger.info(f"Triplet loss: {tri_loss.item():.6f}")
                    logger.info(f"SupCon loss: {supcon_loss.item():.6f}")
                    logger.info(f"Lambda SupCon: {cfg.MODEL.PROMPTSG.LAMBDA_SUPCON if hasattr(cfg.MODEL.PROMPTSG, 'LAMBDA_SUPCON') else 'N/A'}")
                    logger.info(f"Lambda Triplet: {cfg.MODEL.PROMPTSG.LAMBDA_TRIPLET if hasattr(cfg.MODEL.PROMPTSG, 'LAMBDA_TRIPLET') else 'N/A'}")
                    logger.info(f"Lambda ID: {cfg.MODEL.PROMPTSG.LAMBDA_ID if hasattr(cfg.MODEL.PROMPTSG, 'LAMBDA_ID') else 'N/A'}")
                    logger.info("==========================")
                # ============ END DEBUG ============

            scaler.scale(loss).backward()

            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                # cls_score is a list [cls_score, cls_score_proj], use first one for accuracy
                main_cls_score = cls_score[0] if isinstance(cls_score, (list, tuple)) else cls_score
                acc = (main_cls_score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            id_meter.update(id_loss.item(), img.shape[0])
            tri_meter.update(tri_loss.item(), img.shape[0])
            supcon_meter.update(supcon_loss.item(), img.shape[0])
            acc_meter.update(acc.item(), 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                log_msg = ("Epoch[{}] Iteration[{}/{}] Loss: {:.3f} (ID {:.3f} TRI {:.3f} SupCon {:.3f}) Acc: {:.3f} Lr: {:.2e}".format(
                    epoch, n_iter + 1, len(train_loader),
                    loss_meter.avg, id_meter.avg, tri_meter.avg, supcon_meter.avg,
                    acc_meter.avg,
                    scheduler.get_lr()[0] if hasattr(scheduler, 'get_lr') else optimizer.param_groups[0]['lr']
                ))
                logger.info(log_msg)
                metrics_logger.info(log_msg)
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            epoch_msg = "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(epoch, time_per_batch, train_loader.batch_size / time_per_batch)
            logger.info(epoch_msg)
            metrics_logger.info(epoch_msg)

        # Step scheduler AFTER epoch (not before)
        scheduler.step()

        # Collect epoch metrics for CSV
        current_lr = scheduler.get_lr()[0] if hasattr(scheduler, 'get_lr') else optimizer.param_groups[0]['lr']
        epoch_mAP, epoch_r1, epoch_r5, epoch_r10 = '', '', '', ''

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, pid, camid, camids_batch, viewid, img_path) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            feat = model(img)
                            evaluator.update((feat, pid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    val_msg = "Validation Results - Epoch {}".format(epoch)
                    logger.info(val_msg)
                    metrics_logger.info(val_msg)
                    map_msg = "mAP: {:.1%}".format(mAP)
                    logger.info(map_msg)
                    metrics_logger.info(map_msg)
                    for r in [1, 5, 10]:
                        rank_msg = "Rank-{:<3}:{:.1%}".format(r, cmc[r - 1])
                        logger.info(rank_msg)
                        metrics_logger.info(rank_msg)
                    epoch_mAP = f'{mAP:.4f}'
                    epoch_r1 = f'{cmc[0]:.4f}'
                    epoch_r5 = f'{cmc[4]:.4f}'
                    epoch_r10 = f'{cmc[9]:.4f}'
                    torch.cuda.empty_cache()
            else:
                model.eval()
                evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
                evaluator.reset()
                for n_iter, (img, pid, camid, camids_batch, viewid, img_path) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        feat = model(img)
                        evaluator.update((feat, pid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                val_msg = "Validation Results - Epoch {}".format(epoch)
                logger.info(val_msg)
                metrics_logger.info(val_msg)
                map_msg = "mAP: {:.1%}".format(mAP)
                logger.info(map_msg)
                metrics_logger.info(map_msg)
                for r in [1, 5, 10]:
                    rank_msg = "Rank-{:<3}:{:.1%}".format(r, cmc[r - 1])
                    logger.info(rank_msg)
                    metrics_logger.info(rank_msg)
                epoch_mAP = f'{mAP:.4f}'
                epoch_r1 = f'{cmc[0]:.4f}'
                epoch_r5 = f'{cmc[4]:.4f}'
                epoch_r10 = f'{cmc[9]:.4f}'
                torch.cuda.empty_cache()

        # Write epoch row to CSV
        csv_writer.writerow([
            epoch,
            f'{loss_meter.avg:.6f}',
            f'{id_meter.avg:.6f}',
            f'{tri_meter.avg:.6f}',
            f'{supcon_meter.avg:.6f}',
            f'{acc_meter.avg:.4f}',
            f'{current_lr:.2e}',
            epoch_mAP, epoch_r1, epoch_r5, epoch_r10
        ])
        csv_file.flush()

    csv_file.close()
    total_time = time.monotonic() - all_start_time
    time_msg = "Total running time: {:.1f}[s]".format(total_time)
    logger.info(time_msg)
    metrics_logger.info("="*50)
    metrics_logger.info("=== TRAINING COMPLETED ===")
    metrics_logger.info(time_msg)
    metrics_logger.info("="*50)

    # Auto-generate learning curves
    if not cfg.MODEL.DIST_TRAIN or (cfg.MODEL.DIST_TRAIN and dist.get_rank() == 0):
        logger.info("Training completed. Auto-generating learning curves...")
        success = auto_generate_plots(cfg)
        if success:
            logger.info(" Training completed successfully! Check 'plots/' directory for learning curves.")
        else:
            logger.warning("  Plot generation failed. You can manually run: python plot_learning_curves.py")
    else:
        logger.info("Training completed. Plot generation skipped for distributed training.")

    return


def do_inference(cfg, model, val_loader, num_query):
    device = cfg.MODEL.DEVICE
    logger = logging.getLogger("promptsg.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()

    for n_iter, (img, pid, camid, camid_batch, viewid, img_path) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            feat = model(img)
            evaluator.update((feat, pid, camid))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4], mAP


def do_inference_with_visualization(cfg, model, val_loader, num_query, output_dir, num_visualize=20):
    """
    Inference with attention map visualization.

    Args:
        cfg: Config object
        model: PromptSG model
        val_loader: Validation data loader
        num_query: Number of query images
        output_dir: Directory to save visualizations
        num_visualize: Number of images to visualize
    """
    import cv2
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    device = cfg.MODEL.DEVICE
    logger = logging.getLogger("promptsg.test")
    logger.info("Enter inferencing with visualization")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    # Create visualization directory
    vis_dir = os.path.join(output_dir, 'attention_visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    vis_count = 0

    for n_iter, (img, pid, camid, camid_batch, viewid, img_path) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            feat = model(img)
            evaluator.update((feat, pid, camid))

            # Generate visualizations for first N images
            if vis_count < num_visualize:
                # Get model without DataParallel wrapper
                actual_model = model.module if hasattr(model, 'module') else model

                if hasattr(actual_model, 'get_attention_map'):
                    for i in range(min(img.size(0), num_visualize - vis_count)):
                        try:
                            # Get attention map
                            single_img = img[i:i+1]
                            attn_map = actual_model.get_attention_map(single_img, reshape_to_image=True)
                            attn_map = attn_map.cpu().numpy()[0]

                            # Normalize
                            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

                            # Resize to image size
                            h, w = cfg.INPUT.SIZE_TEST
                            attn_map_resized = cv2.resize(attn_map, (w, h))

                            # Denormalize image
                            mean = [0.485, 0.456, 0.406]
                            std = [0.229, 0.224, 0.225]
                            img_np = single_img[0].cpu().clone()
                            for c, (m, s) in enumerate(zip(mean, std)):
                                img_np[c] = img_np[c] * s + m
                            img_np = img_np.clamp(0, 1).permute(1, 2, 0).numpy()
                            img_np = np.uint8(255 * img_np)

                            # Create heatmap
                            heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_JET)
                            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                            # Create overlay
                            overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

                            # Save visualization
                            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                            axes[0].imshow(img_np)
                            pid_val = pid[i].item() if hasattr(pid[i], 'item') else pid[i]
                            axes[0].set_title(f'PID: {pid_val}')
                            axes[0].axis('off')

                            axes[1].imshow(attn_map_resized, cmap='jet')
                            axes[1].set_title('Attention Map')
                            axes[1].axis('off')

                            axes[2].imshow(overlay)
                            axes[2].set_title('Overlay')
                            axes[2].axis('off')

                            plt.tight_layout()
                            pid_val = pid[i].item() if hasattr(pid[i], 'item') else pid[i]
                            save_path = os.path.join(vis_dir, f'vis_{vis_count}_pid{pid_val}.png')
                            plt.savefig(save_path, dpi=150, bbox_inches='tight')
                            plt.close()

                            vis_count += 1

                        except Exception as e:
                            logger.warning(f"Failed to visualize image {vis_count}: {e}")

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    logger.info(f"Saved {vis_count} attention visualizations to {vis_dir}")

    return cmc[0], cmc[4], mAP