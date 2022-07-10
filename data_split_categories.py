import pdb
import pandas as pd
import shutil
import os

full_df = pd.read_csv('full_df.csv')
keep = ['normal_fundus', 'moderate non proliferative retinopathy',
        'mild nonproliferative retinopathy', 'cataract', 'pathological myopia',
        'glaucoma', 'dry age-related macular degeneration',
        'severe nonproliferative retinopathy', 'dry age-related macular degeneration']
keep = [x.replace(' ', '_') for x in keep]

for i, row in full_df.iterrows():
	label_left = row['Left-Diagnostic Keywords']
	label_right = row['Right-Diagnostic Keywords']

	## figure out if the image is left or right
	if 'right' in row['filename']:
		label = label_right
	elif 'left' in row['filename']:
		label = label_left
	else:
		exit('error')
	
	label = label.replace(' ', '_')
	if label not in keep: continue
	os.makedirs('data/'+label, exist_ok=True)
	source = 'preprocessed_images/'+row['filename']
	destination = 'data/'+label+'/'+row['filename']
	shutil.copy(source, destination)