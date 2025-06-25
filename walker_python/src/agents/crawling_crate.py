# Main chassis fixture
chassis_shape = b2.b2PolygonShape(box=(1.5, 0.75))
chassis_fixture = self.body.CreateFixture(
    shape=chassis_shape, 
    density=4.0, 
    friction=0.8,
    filter=b2.b2Filter(categoryBits=category_bits, maskBits=mask_bits)
)
# chassis_fixture.filterData = self.filter  # FIXED: Use proper collision filtering

# Wheel fixtures
wheel_fixture = wheel.CreateFixture(
    shape=b2.b2CircleShape(radius=0.5), 
    density=8.0, 
    friction=0.9,
    filter=b2.b2Filter(categoryBits=category_bits, maskBits=mask_bits)
)
# wheel_fixture.filterData = self.filter  # FIXED: Use proper collision filtering

# Upper arm fixture
upper_arm_fixture = self.upper_arm.CreateFixture(
    shape=b2.b2PolygonShape(box=(1.25, 0.1)), 
    density=0.1,  # Very light arms for better control
    friction=0.5,
    filter=b2.b2Filter(categoryBits=category_bits, maskBits=mask_bits)
)
# upper_arm_fixture.filterData = self.filter  # FIXED: Use proper collision filtering

# Lower arm fixture
lower_arm_fixture = self.lower_arm.CreateFixture(
    shape=b2.b2PolygonShape(box=(1.25, 0.1)), 
    density=0.1,  # Very light arms for better control
    friction=0.5,  # Reduced friction to prevent sticking
    filter=b2.b2Filter(categoryBits=category_bits, maskBits=mask_bits)
)
# lower_arm_fixture.filterData = self.filter  # FIXED: Use proper collision filtering 