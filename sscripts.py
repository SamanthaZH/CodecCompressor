import gin
import subprocess
import os

def run_scenes(start_scene, end_scene): #base_command, config_path_template):
    base_command = "python main.py"
    config_path_template = "config_files/batch/tricp_f{}.gin"
    for scene_number in range(start_scene, end_scene + 1):
        scene_config_file = config_path_template.format(scene_number)
        command = f"{base_command} --ginc {scene_config_file}"
        print(f"==>>Executing f{scene_number}: {command}")
        subprocess.run(command, shell=True)

def generate_gin_files(nm,base_config_path, output_folder, start_scene=2, end_scene=60):
    # Generate gin files script
    os.makedirs(output_folder,exist_ok = True)
    with open(base_config_path, 'r') as file:
        base_config = file.read()

    for i in range(start_scene, end_scene + 1):
        new_scene = f'{nm}_f{i}'
        new = f'f{i}'
        # Replace the scene in the base configuration
        modified_config = base_config.replace('lego_f3', new_scene)

        # Write the modified configuration to a new file
        output_path = f'{output_folder}/tricp_{new}.gin'
        with open(output_path, 'w') as file:
           file.write(modified_config)
        print(f'Generated configuration for scene {new_scene} at {output_path}')

def generate_testgin_files(base_config_path, output_folder, start_scene=2, end_scene=5):
    # Generate gin files script
    os.makedirs(output_folder,exist_ok = True)
    with open(base_config_path, 'r') as file:
        base_config = file.read()

    for i in range(start_scene, end_scene + 1):
        new_scene = f'worker_f{i}'
        new = f'f{i}'
        new_model_dir = [item for item in os.listdir('./log_worker/fd1/nerf_synthetic/'+new_scene+'/Tri-MipRF/')]
        print(new_model_dir[0])
        # Replace the scene in the base configuration
        modified_config = base_config.replace('lego_f5', new_scene)
        modified_config = modified_config.replace('2024-07-10_15-32-51', new_model_dir[0])
        # Write the modified configuration to a new file
        output_path = f'{output_folder}/tricp_{new}.gin'
        print(f'Generated configuration for scene {new_scene} at {output_path}')
        with open(output_path, 'w') as file:
           file.write(modified_config)
        

def run_test(start_scene, end_scene): #base_command, config_path_template):
    base_command = "python test_model.py"
    config_path_template = "config_files/test/tricp_f{}.gin"
    for scene_number in range(start_scene, end_scene + 1):
        scene_config_file = config_path_template.format(scene_number)
        command = f"{base_command} --ginc {scene_config_file}"
        print(f"==>>Executing f{scene_number}: {command}")
        subprocess.run(command, shell=True)

generate_gin_files('pig','./config_files/blender/f5.gin', './config_files/batch')
#run_scenes(24,29)
#generate_testgin_files('./config_files/blender/test_only_f5.gin', './config_files/test')
#run_test(2,5)