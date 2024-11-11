import torch
import numpy as np
import cv2
from raft.core.raft import RAFT
from raft.core.utils import flow_viz

def estimate_flow(img1_path, img2_path, output_path):
  model = RAFT(cfg=RAFTConfig(
      small=True,  # Use small model for faster inference (optional)
      alternate_corr=False,  # Use standard correlation (optional)
  ))
  model.load_state_dict(torch.load('raft-sintel.pth'))
  model.eval()

  # Load images
  img1 = cv2.imread(img1_path)
  img2 = cv2.imread(img2_path)
  img1 = torch.from_numpy(img1).permute(2, 0, 1).float().unsqueeze(0) / 255.0
  img2 = torch.from_numpy(img2).permute(2, 0, 1).float().unsqueeze(0) / 255.0
  output_path = 'raft-'+img1[:4]+'_'+img2[:4]+'.png'


  with torch.no_grad():
      flow_low, flow_up = model(img1, img2, iters=20, test_mode=True)

  flow = flow_up[0].permute(1, 2, 0).cpu().numpy()
  flow_image = flow_viz.flow_to_image(flow)
  cv2.imwrite(output_path, flow_image)




img1_path = input("Enter path to first image: ")
img2_path = input("Enter path to second image: ")
estimate_flow(img1_path, img2_path)
print(f"output saved to: {output_path}")