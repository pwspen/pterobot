import xml.etree.ElementTree as ET

# This is a janky workaround because I didn't know how to use/load keyframes - fix!

# MJDATA.TXT is exported from Mujoco Simulator and contains qpos and qvel
mjdata_file = 'MJDATA.TXT'

# XML file to adjust must already have entries for qpos and qvel in it somewhere.
# Updated_pose will be created from origin file - origin file not modified.
xml_origin = 'load/pterobot_v0.xml'
xml_updated_pose = xml_origin.replace('.xml', '_updated_pose.xml')

# If it doesn't have qpos/qvel, paste this into it:
# <custom>
# <numeric name="init_qpos" data="" />
# <numeric name="init_qvel" data="" />
# </custom>

# with open(xml_to_adjust_pose, 'r', encoding='cp1252') as file:
#     for line in file:
#         print(line)

with open(mjdata_file, 'r') as file:
    qposmode, qvelmode = False, False
    vals = []
    qpos, qvel = None, None
    for line in file:
        if qposmode or qvelmode:
            if not (line == '' or line == '\n'):
                txt = line.replace('\n', '')
                txt = txt.strip()
                txt = float(txt)
                txt = 0 if abs(txt) < 0.001 else txt
                vals.append(txt)
            else:
                if qposmode:
                    qpos = vals
                    qposmode = False
                    vals = []
                elif qvelmode:
                    qvel = vals
                    qvelmode = False
                    vals = []

        if 'QPOS' in line:
            qposmode = True

        elif 'QVEL' in line:
            qvelmode = True
            
    print(f'self.user_init_qpos = np.array({qpos})')
    print(f'self.user_init_qvel = np.array({qvel})\n')

    assert qpos is not None, 'qpos not found in mjdata file'
    assert qvel is not None, 'qvel not found in mjdata file'

    qpos, qvel = ' '.join(qpos), ' '.join(qvel)

tree = ET.parse(xml_origin)
root = tree.getroot()

# Iterate through custom/numeric elements and update the 'data' attribute based on the 'name'
for numeric in root.findall(".//custom/numeric"):
    name = numeric.get('name')
    if name == "init_qvel":
        numeric.set('size', str(len(qvel.split(' '))))
        numeric.set('data', qvel)
    elif name == "init_qpos":
        numeric.set('size', str(len(qpos.split(' '))))
        numeric.set('data', qpos)

tree.write(xml_updated_pose)
print(f'Pose saved to {xml_updated_pose}')


#<custom>
#   <numeric name="init_qvel" size="17" data="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"/>
#   <numeric name="init_qpos" size="17" data="0 0 0.55 1 0 0 0 0 1 0 0 0 1 0 0 0 1"/>
#</custom>

