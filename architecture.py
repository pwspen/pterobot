limb_linear_density = 1 # kg/m

# head, neck
head_length = 0.1 # m
head_mass = 0.5 # kg
neck_length = 0.1 # m

head_and_neck_mass = head_mass + neck_length * limb_linear_density

# wings
humerus_length = 0.3 # m
radius_length = 0.2 # m

wing_mass = (humerus_length + radius_length) * limb_linear_density

# legs
femur_length = 0.4 # m
tibia_length = 0.3 # m

leg_mass = (femur_length + tibia_length) * limb_linear_density

