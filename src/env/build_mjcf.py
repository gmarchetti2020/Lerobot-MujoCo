import xml.etree.ElementTree as ET
from src.mujoco_helper.utils import prettify
import random
import os

def build_mjcf_based_on_config(base_xml_path,recp_names, obj_names):
    """
    This function is used to build a mjcf file based on the config file.
    The config file is a list of dictionaries, each dictionary contains the following keys:
    - language_instruction: str
    - conditions: dict
        - objects: list of dict
            - names: tuple
            - relation: str
        - gripper: dict
            - state: str
            - pose: tuple
    - xml_file: str
    """
    # Resolve the base XML directory — temp.xml will be written here so that
    # all relative paths (meshdir, includes) resolve correctly.
    base_xml_dir = os.path.dirname(os.path.abspath(base_xml_path))
    # Directory containing recp/ and objects/ model_new.xml files
    asset_dir = os.path.abspath("./asset")

    # Load the base xml file
    tree = ET.parse(base_xml_path)
    root = tree.getroot()

    # Existing <include> elements keep their original relative paths since
    # temp.xml will be written to the same directory as the base XML.

    # Add receptacles — use absolute paths since these files live in a
    # different directory tree (asset/) than the base XML.
    # Only add includes for model files that actually exist on disk.
    for recp_name in recp_names:
        file_path = os.path.normpath(os.path.join(asset_dir, 'recp', recp_name, 'model_new.xml'))
        if os.path.exists(file_path):
            include_tag = ET.Element('include',attrib={'file':file_path})
            root.append(include_tag)
    for obj_name in obj_names:
        file_path = os.path.normpath(os.path.join(asset_dir, 'objects', obj_name, 'model_new.xml'))
        if os.path.exists(file_path):
            include_tag = ET.Element('include',attrib={'file':file_path})
            root.append(include_tag)

    xml_string = prettify(root) # indent xml
    # Write temp.xml next to the base XML so MuJoCo resolves meshdir and
    # other relative paths from the same directory as the original scene.
    xml_path = os.path.join(base_xml_dir, "temp.xml")
    with open(xml_path,'w') as f:
        f.write(xml_string)
    return xml_path
