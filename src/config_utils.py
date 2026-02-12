import re
import os
import shutil
import subprocess
from src.general_utils import replace_next_line


def read_config_file(config_file):
    # Dictionary to store parameter names and values
    parameters = {}

    with open(config_file, "r") as file:
        # Read lines from the file
        lines = file.readlines()

        # Define a regular expression pattern to match parameter lines
        pattern = re.compile(r"\s*([^#%]+)\s*=\s*([^#%]+)\s*")

        # Iterate through lines and extract parameters
        for line in lines:
            # Remove content after '%' symbol
            line = line.split("%")[0].strip()

            # Skip lines starting with '%'
            if line.startswith("%"):
                continue

            match = pattern.match(line)
            if match:
                parameter_name = match.group(1).strip()
                parameter_value = match.group(2).strip()
                parameters[parameter_name] = parameter_value

    return parameters


def generate_log_filename(parameters):
    log_dir = parameters.get("LOG_DIR", "")
    log_file = parameters.get("LOG_FILE", "")

    # Concatenate log_dir and log_file to form the resulting filename
    if log_dir and log_file:
        log_filename = f"{log_dir}/{log_file}"
        return log_filename
    else:
        print("Error: LOG_DIR or LOG_FILE not found in the parameters.")
        return None


def update_parameter(parameters, key, new_value):
    # Create a new dictionary to avoid modifying the original parameters
    updated_parameters = parameters.copy()

    # Update the specified key with the new value
    updated_parameters[key] = new_value

    return updated_parameters


def write_config_file(parameters, output_file_path):
    with open(output_file_path, "w") as file:
        file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        file.write("%  Auto-generated Config File           %\n")
        file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n")

        for key, value in parameters.items():
            file.write(f"{key} = {value}\n")


def remove_files(filename):
    # Construct file paths for the log file and CSV file
    log_file_path = filename
    csv_file_path = filename + ".csv"

    # Remove the log file if it exists
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
        print(f"Removed {log_file_path}")

    # Remove the CSV file if it exists
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)
        print(f"Removed {csv_file_path}")


def update_quarter_hohlraum_mesh_file(n_cell, filepath):
    filename_geo = filepath + "quarter_hohlraum.geo"
    filename_geo_backup = filepath + "sym_hohlraum_backup.geo"

    filename_su2 = filepath + f"quarter_hohlraum_p{n_cell}.su2"
    filename_vtk = filepath + f"quarter_hohlraum_p{n_cell}.vtk"
    filename_con = filepath + f"quarter_hohlraum_p{n_cell}.con"

    if not os.path.exists(filename_su2):
        shutil.copy(filename_geo, filename_geo_backup)

        with open(filename_geo_backup, "r") as file:
            lines = file.readlines()

        with open(filename_geo_backup, "w") as file:
            for line in lines:
                if line.startswith("cl_fine"):
                    line = f"cl_fine = {n_cell};\n"
                file.write(line)

        # Remove the .con file
        if os.path.exists(filename_con):
            os.remove(filename_con)

        os.system(
            f"gmsh {filename_geo_backup} -2 -format su2 -save_all -o {filename_su2}"
        )
        os.system(
            f"gmsh {filename_geo_backup} -2 -format vtk -save_all -o {filename_vtk}"
        )
        os.remove(filename_geo_backup)

    return f"quarter_hohlraum_p{n_cell}.su2"


def update_sym_hohlraum_mesh_file(n_cell, filepath):
    filename_geo = filepath + "sym_hohlraum.geo"
    filename_geo_backup = filepath + "sym_hohlraum_backup.geo"

    filename_su2 = filepath + f"sym_hohlraum_n{n_cell}.su2"
    filename_vtk = filepath + f"sym_hohlraum_n{n_cell}.vtk"
    filename_con = filepath + f"sym_hohlraum_n{n_cell}.con"

    if not os.path.exists(filename_su2):
        shutil.copy(filename_geo, filename_geo_backup)

        with open(filename_geo_backup, "r") as file:
            lines = file.readlines()

        with open(filename_geo_backup, "w") as file:
            for line in lines:
                if line.startswith("n_coarse_recombine"):
                    line = f"n_coarse_recombine = {n_cell};\n"
                file.write(line)

        # Remove the .con file
        if os.path.exists(filename_con):
            os.remove(filename_con)

        print("saving mesh with n_cell = ", n_cell)
        os.system(
            f"gmsh {filename_geo_backup} -2 -format su2 -save_all -o {filename_su2}"
        )
        os.system(
            f"gmsh {filename_geo_backup} -2 -format vtk -save_all -o {filename_vtk}"
        )
        os.remove(filename_geo_backup)

    return f"sym_hohlraum_n{n_cell}.su2"


def update_var_hohlraum_mesh_file(
    hpc_mode,
    filepath,
    cl_fine,
    upper_left_red,
    lower_left_red,
    upper_right_red,
    lower_right_red,
    horizontal_left_red,
    horizontal_right_red,
    capsule_x,
    capsule_y,
):
    filename_geo = filepath + "hohlraum_variable.geo"
    unique_name = f"hohlraum_variable_cl{cl_fine}_ulr{upper_left_red}_llr{lower_left_red}_urr{upper_right_red}_lrr{lower_right_red}_hlr{horizontal_left_red}_hrr{horizontal_right_red}_cx{capsule_x}_cy{capsule_y}"
    filename_geo_backup = filepath + "backup_" + unique_name + ".geo"
    filename_su2 = filepath + unique_name + ".su2"
    filename_vtk = filepath + unique_name + ".vtk"
    filename_con = filepath + unique_name + ".con"

    if not os.path.exists(filename_su2):
        shutil.copy(filename_geo, filename_geo_backup)

        with open(filename_geo_backup, "r") as file:
            lines = file.readlines()

        with open(filename_geo_backup, "w") as file:
            for line in lines:
                if line.startswith("cl_fine"):
                    line = f"cl_fine = {cl_fine};\n"
                if line.startswith("upper_left_red"):
                    line = f"upper_left_red = {upper_left_red};\n"
                if line.startswith("lower_left_red"):
                    line = f"lower_left_red = {lower_left_red};\n"
                if line.startswith("upper_right_red"):
                    line = f"upper_right_red = {upper_right_red};\n"
                if line.startswith("lower_right_red"):
                    line = f"lower_right_red = {lower_right_red};\n"
                if line.startswith("horizontal_left_red"):
                    line = f"horizontal_left_red = {horizontal_left_red};\n"
                if line.startswith("horizontal_right_red"):
                    line = f"horizontal_right_red = {horizontal_right_red};\n"
                if line.startswith("capsule_x"):
                    line = f"capsule_x = {capsule_x};\n"
                if line.startswith("capsule_y"):
                    line = f"capsule_y = {capsule_y};\n"
                file.write(line)

        # Remove the .con file
        if os.path.exists(filename_con):
            os.remove(filename_con)

        print("saving mesh with cl = ", cl_fine)
        # if hpc_mode:
        #
        #    basic_slurm_file = "./slurm_template.sh"
        #
        #    # Read the input file
        #    with open(basic_slurm_file, "r") as file:
        #        lines = file.readlines()
        #    # Replace the last line
        #    if lines:
        #        lines[-1] = (
        #            f"source venv/bin/activate\n gmsh {filename_geo_backup} -2 -format su2 -save_all -o {filename_su2}\n"
        #        )
        #
        #    slurm_script_path = filepath + "gmsh_job.sh"
        #
        #    with open(slurm_script_path, "w") as file:
        #        file.writelines(lines)
        #
        #    # Submit SLURM job
        #    subprocess.run(["sbatch", slurm_script_path])
        #    # Remove SLURM job script
        #    # os.remove(slurm_script_path)
        #
        # else:
        os.system(
            f"gmsh {filename_geo_backup} -2 -format su2 -save_all -o {filename_su2}"
        )
        # os.system(f"gmsh {filename_geo_backup} -2 -format vtk -save_all -o {filename_vtk}")
        os.remove(filename_geo_backup)
    return unique_name + ".su2"


def update_var_quarter_hohlraum_mesh_file(
    hpc_mode,
    filepath,
    cl_fine,
    upper_right_red,
    horizontal_right_red,
    rectangular_mesh=False,
):
    if rectangular_mesh:
        filename_geo = (
            filepath + "quarter_hohlraum_rectangular.geo"
        )  # "quarter_hohlraum_rectangular.geo"
    else:
        filename_geo = (
            filepath + "quarter_hohlraum_variable.geo"
        )  # "quarter_hohlraum_rectangular.geo"
    unique_name = (
        f"hohlraum_variable_cl{cl_fine}_urr{upper_right_red}_hrr{horizontal_right_red}"
    )
    filename_geo_backup = filepath + "backup_" + unique_name + ".geo"
    filename_su2 = filepath + unique_name + ".su2"
    filename_vtk = filepath + unique_name + ".vtk"
    filename_con = filepath + unique_name + ".con"

    if not os.path.exists(filename_su2):
        shutil.copy(filename_geo, filename_geo_backup)

        with open(filename_geo_backup, "r") as file:
            lines = file.readlines()

        with open(filename_geo_backup, "w") as file:
            for line in lines:
                if line.startswith("cl_fine"):
                    line = f"cl_fine = {cl_fine};\n"
                if line.startswith("upper_right_red"):
                    line = f"upper_right_red = {upper_right_red};\n"
                if line.startswith("horizontal_right_red"):
                    line = f"horizontal_right_red = {horizontal_right_red};\n"
                file.write(line)

        # Remove the .con file
        if os.path.exists(filename_con):
            os.remove(filename_con)

        print("saving mesh with cl = ", cl_fine)
        # if hpc_mode:
        #
        #    basic_slurm_file = "./slurm_template.sh"
        #
        #    # Read the input file
        #    with open(basic_slurm_file, "r") as file:
        #        lines = file.readlines()
        #    # Replace the last line
        #    if lines:
        #        lines[-1] = (
        #            f"source venv/bin/activate\n gmsh {filename_geo_backup} -2 -format su2 -save_all -o {filename_su2}\n"
        #        )
        #
        #    slurm_script_path = filepath + "gmsh_job.sh"
        #
        #    with open(slurm_script_path, "w") as file:
        #        file.writelines(lines)
        #
        #    # Submit SLURM job
        #    subprocess.run(["sbatch", slurm_script_path])
        #    # Remove SLURM job script
        #    os.remove(slurm_script_path)
        #
        # else:
        os.system(
            f"gmsh {filename_geo_backup} -2 -format su2 -save_all -o {filename_su2}"
        )
        # os.system(f"gmsh {filename_geo_backup} -2 -format vtk -save_all -o {filename_vtk}")
        os.remove(filename_geo_backup)
    return unique_name + ".su2"


def update_lattice_mesh_file(n_cell, filepath, rectangular_mesh=False):
    filename_geo = filepath + "lattice_triangular.geo"
    filename_geo_backup = filepath + "lattice_triangular_backup.geo"

    filename_su2 = filepath + f"lattice_p{n_cell}.su2"
    filename_con = filepath + f"lattice_p{n_cell}.con"

    if not os.path.exists(filename_su2):
        shutil.copy(filename_geo, filename_geo_backup)

        with open(filename_geo_backup, "r") as file:
            lines = file.readlines()

        with open(filename_geo_backup, "w") as file:
            for line in lines:
                if line.startswith("cl_fine"):
                    line = f"cl_fine = {n_cell};\n"
                file.write(line)

        # Remove the .con file
        if os.path.exists(filename_con):
            os.remove(filename_con)

        os.system(
            f"gmsh {filename_geo_backup} -2 -format su2 -save_all -o {filename_su2}"
        )
        os.remove(filename_geo_backup)

    return f"lattice_p{n_cell}.su2"


def update_half_lattice_mesh_file(n_cell, filepath, rectangular_mesh=False):
    if rectangular_mesh:
        filename_geo = filepath + "half_lattice_rectangular.geo"
    else:
        filename_geo = filepath + "half_lattice_homogeneous.geo"
    filename_geo_backup = filepath + "half_lattice_backup.geo"
    filename_su2 = filepath + f"half_lattice_p{n_cell}.su2"
    filename_vtk = filepath + f"half_lattice_p{n_cell}.vtk"
    filename_con = filepath + f"half_lattice_p{n_cell}.con"

    if not os.path.exists(filename_su2):
        shutil.copy(filename_geo, filename_geo_backup)

        with open(filename_geo_backup, "r") as file:
            lines = file.readlines()

        with open(filename_geo_backup, "w") as file:
            for line in lines:
                if line.startswith("cl_fine"):
                    line = f"cl_fine = {n_cell};\n"
                file.write(line)

        # Remove the .con file
        if os.path.exists(filename_con):
            os.remove(filename_con)

        os.system(
            f"gmsh {filename_geo_backup} -2 -format su2 -save_all -o {filename_su2}"
        )
        # os.system(
        #    f"gmsh {filename_geo_backup} -2 -format vtk -save_all -o {filename_vtk}"
        # )
        os.remove(filename_geo_backup)

    return f"half_lattice_p{n_cell}.su2"


def write_slurm_file(
    output_slurm_dir, unique_name, subfolder, singularity=True, use_cuda=False
):
    if use_cuda:
        raise ValueError(
            "CUDA mode with SLURM is not supported in this workflow."
        )
    basic_slurm_file = "./slurm_template.sh"

    # Ensure the output directory exists
    if not os.path.exists(output_slurm_dir):
        os.makedirs(output_slurm_dir)

    # Read the input file
    with open(basic_slurm_file, "r") as file:
        lines = file.readlines()

    # Replace the last line
    if lines:
        if singularity:
            lines[-1] = (
                "singularity exec KiT-RT/tools/singularity/kit_rt.sif ./KiT-RT/build_singularity/KiT-RT "
                + subfolder
                + unique_name
                + ".cfg\n"
            )

        else:
            lines[-1] = "srun ./KiT-RT/build/KiT-RT " + subfolder + unique_name + ".cfg\n"

    # Write the modified lines to the output file
    with open(output_slurm_dir + unique_name + ".sh", "w") as file:
        file.writelines(lines)

    return 0


def read_username_from_config(config_file):
    """
    Read the username from the configuration file.
    The file should have a line in the format USER=<username>.
    """
    try:
        with open(config_file, "r") as file:
            for line in file:
                if line.startswith("USER="):
                    return line.strip().split("=")[1]
    except Exception as e:
        print(f"Error reading config file: {e}")
    return None
