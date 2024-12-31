#include "led_ring.hpp"

#include <gz/plugin/Register.hh>
// Register the plugin with Gazebo Fortress
IGNITION_ADD_PLUGIN(
    gzplugin::LedRingPlugin,
    gz::sim::System,
	gzplugin::LedRingPlugin::ISystemConfigure, //)
	gzplugin::LedRingPlugin::ISystemPreUpdate)

// Optional alias for the plugin
IGNITION_ADD_PLUGIN_ALIAS(gzplugin::LedRingPlugin, "gzplugin::LedRingPlugin")


#define PI 3.14159265358979323846

// using namespace ignition;
namespace gzplugin
{
	
// Constructor
LedRingPlugin::LedRingPlugin(){}

// Destructor
LedRingPlugin::~LedRingPlugin() {}

// void LedRingPlugin::Load(gz::sim::EntityComponentManager &_ecm, sdf::ElementPtr _sdf) {
  // Implement Configure callback, provided by ISystemConfigure
  // and called once at startup.
void LedRingPlugin::Configure(const gz::sim::Entity &_entity,
				const std::shared_ptr<const sdf::Element> &_sdf,
				gz::sim::EntityComponentManager &_ecm,
				gz::sim::EventManager &_eventMgr)
{

	// // Create a GazeboRos node instead of a common ROS node.
	// // Pass it SDF parameters so common options like namespace and remapping
	// // can be handled. 
	// Not needed anymore as ROS not integrated as closely
	// ros_node_ = gazebo_ros::Node::Get(_sdf);

	this->ecm = &_ecm;
	this->em = &_eventMgr;

	// // Obtain the model entity (model pointer is no longer passed directly)
    // gz::sim::Entity modelEntity = gz::sim::Model(_sdf->GetParent());
    this->model = gz::sim::Model(_entity); //std::make_shared<gz::physics::Model>(modelEntity);
	if(!this->model.Valid(_ecm))
	{
		ignerr << "LedRingPlugin should be attached to a model entity, failed to initialise" << std::endl;
		return;
	}

	// World is initialised in the postupdate

	// // Get Physics	
	// this->physics = this->world->Physics();

	// // // Get parameters specified in the sdf file.
	if (_sdf->HasElement("robotNamespace")) {
		this->robot_namespace = _sdf->Get<std::string>("robotNamespace");
	}
	if (_sdf->HasElement("ring_z_offset")) {
		this->ring_z_offset = _sdf->Get<double>("ring_z_offset");
	}
	if (_sdf->HasElement("ring_radius")) {
		this->ring_radius = _sdf->Get<double>("ring_radius");
	}
	if (_sdf->HasElement("n_leds")) {
		this->n_leds = _sdf->Get<double>("n_leds");
	}
	
	// Initialise Led Colours
	for(int i = 0; i < this->n_leds * 3; i++) {
		this->led_colours.push_back(0.0);
	}

	// // Create Publisher and Subscriber for ROS2 to interact with this
	// this->publisher_name = this->robot_namespace + "/leds/status";
	// this->led_status_pub = node.Advertise<ignition::msgs::Float_V>(this->publisher_name);
	// if (!this->led_status_pub)
	// {
	// 	ignerr << "Error advertising topic [" << this->publisher_name << "]" << std::endl;
	// 	return;
	// }

	// this->subscription_name = this->robot_namespace+"/leds/control";
	// if(!this->node.Subscribe(this->subscription_name, &LedRingPlugin::OnLedMsg, this))
	// {
	// 	ignerr << "Error subscribing to topic [" << this->subscription_name << "]" << std::endl;
	// 	return;
	// }

	// ignerr << "Publisher initialised on " << this->subscription_name << std::endl;
	// ignerr << "Subscriber initialised on " << this->publisher_name<< std::endl;
}

void LedRingPlugin::OnLedMsg(const ignition::msgs::Float_V &leds) {
	auto data = leds.data();
	ignerr << "Received an LED Control msg" << std::endl;
	int led_index = 0;
	for(float d: data) {
		if(led_index > this->led_colours.size()) {
			ignerr << "Received more led colours than expected" << std::endl;
			break;
		}
		ignerr << led_index << " " << std::to_string(d) + " " << std::endl;
		this->led_colours[led_index] = d;
		led_index++;
	}
	this->_received = true;
}

void LedRingPlugin::PreUpdate(const gz::sim::UpdateInfo &_info, gz::sim::EntityComponentManager &_ecm)
{
    // Your logic to check distance or conditions for attaching/detaching joints
    // Example: if (distance < threshold) { attachJoint(); }
	// ignmsg << "SampleSystem::PostUpdate" << std::endl;

	if (!this->_configured) {

		// Set World here as ECM is not fully initialised
		this->world = gz::sim::World(gz::sim::worldEntity(_ecm));
		this->base_link = gz::sim::Link(this->model.CanonicalLink(_ecm));


		gz::sim::SdfEntityCreator entityCreator(_ecm, *(this->em));

		// Construct the LED RING
		// Create a new link
		sdf::Link sdfLink;
		sdfLink.SetName("dynamic_cylinder_link");
		sdfLink.SetPoseRelativeTo("base_frame");
		sdfLink.SetRawPose(gz::math::Pose3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));

		// Set very small mass and inertial properties
		sdfLink.SetInertial(
			gz::math::Inertial(
				gz::math::MassMatrix3d(1e-6, gz::math::Vector3d(1e-12, 1e-12, 1e-12), 
										gz::math::Vector3d(0.0, 0.0, 0.0)),
				gz::math::Pose3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
			)
		);

		double angle_delta = 2*PI/this->n_leds;
		for(int i = 0; i < this->n_leds; i++) {
			// Create a visual
			sdf::Visual sdfVisual;
			sdfVisual.SetName("led" + std::to_string(i));
			sdfVisual.SetPoseRelativeTo("base_frame");

			// Calculate x, y position
			float x = this->ring_radius * std::cos(angle_delta * i);
			float y = this->ring_radius * std::sin(angle_delta * i);
			sdfVisual.SetRawPose(gz::math::Pose3d(x, y, this->ring_z_offset, 0.0, 0.0, 0.0));

			// Set cylinder geometry
			sdf::Cylinder cyl;
			cyl.SetRadius(0.015);
			cyl.SetLength(0.015);
			sdf::Geometry sdfGeom;
			sdfGeom.SetType(sdf::GeometryType::CYLINDER);
			sdfGeom.SetCylinderShape(cyl);
			sdfVisual.SetGeom(sdfGeom);

			float r = this->led_colours[i * 3 + 0];
			float b = this->led_colours[i * 3 + 1];
			float g = this->led_colours[i * 3 + 2];

			// Set material
			sdf::Material sdfMaterial;
			// Set the diffuse color to match the LED's body color
			sdfMaterial.SetDiffuse(gz::math::Color(r,g,b, 1.0));
			// Set the emissive color to simulate light emission
			sdfMaterial.SetEmissive(gz::math::Color(r,g,b, 1.0));  // Bright red
			// Set the specular color to simulate glossy reflection
			sdfMaterial.SetSpecular(gz::math::Color(0.8, 0.8, 0.8, 1.0));  // Near-white for strong highlights
			// Optionally add transparency for a translucent effect
			sdfVisual.SetTransparency(0.1);  // Slightly translucent
			sdfVisual.SetMaterial(sdfMaterial);

			// Add the visual to the link
			sdfLink.AddVisual(sdfVisual);


			// Create a light
			// sdf::Light sdfLight;
			// sdfLight.SetName("led" + std::to_string());
			// sdfLight.SetType(sdf::LightType::POINT);
			// sdfLight.SetDiffuse(gz::math::Color(this->colorR, this->colorG, this->colorB, 1.0));
			// sdfLight.SetSpecular(gz::math::Color(0.8, 0.8, 0.8, 1.0));  // Near-white for strong highlights
			// sdfLight.SetIntensity(1.0);
			// sdfLight.SetAttenuationRange(1.0);
			// sdfLight.SetDirection(gz::math::Vector3d(0.0, 0.0, -1.0));
			// sdfLight.SetRawPose(gz::math::Pose3d(x, y, this->ring_z_offset, 0.0, 0.0, 0.0));

			// Add the light to the link
			sdfLink.AddLight(sdfLight);

		}


		// Create the entities using the EntityCreator
		auto linkEntity = entityCreator.CreateEntities(&sdfLink);
		if(linkEntity == gz::sim::kNullEntity) {
			ignerr << "LinkEntity failed to be created" << std::endl;
		} 
		entityCreator.SetParent(linkEntity, this->model.Entity());
		this->led_link = gz::sim::Link(linkEntity);

		// OPTIONAL: Add a fixed joint to rigidly attach the link to the model
		sdf::Joint sdfJoint;
		sdfJoint.SetName("dynamic_cylinder_joint");
		sdfJoint.SetType(sdf::JointType::FIXED);
		sdfJoint.SetParentLinkName("base_link");
		sdfJoint.SetChildLinkName("dynamic_cylinder_link");

		auto jointEntity = entityCreator.CreateEntities(&sdfJoint, true); // Needs true as it avoids dynamic resolution which will fail to find the base_link
		if(jointEntity == gz::sim::kNullEntity) {
			ignerr << "jointEntity failed to be created" << std::endl;
		}
		entityCreator.SetParent(jointEntity, this->model.Entity());


		this->_configured = true;
		ignerr << "LedRingPlugin Configured"<< std::endl;
		ignmsg << "LedRingPlugin Configured"<< std::endl;
	}

	// Iterate over all visual entities
	_ecm.Each<gz::sim::components::Visual, gz::sim::components::Name>(
	[&](const gz::sim::Entity &entity,
		const gz::sim::components::Visual *,
		const gz::sim::components::Name *name) -> bool
	{
		// Check if the visual is one of the LEDs we created
		if (name->Data().find("led") != std::string::npos)
		{
			std::string led_name = name->Data();

			// Get the Material component
			auto materialComp = _ecm.Component<gz::sim::components::Material>(entity);

			if (materialComp)
			{
				// Get the material message
				gz::math::Color diffuse = materialComp->Data().Diffuse();
				ignerr << "Diffuse State: " << diffuse.R() << " " << diffuse.G() << " " << diffuse.B() << " " << std::endl;
			}
		}			
		return true;  // Continue iteration
	});

	if (this->_received){
		this->_received = false;
		// Change Colour Based On What Has Been Received
		if(this->led_link.Entity() == gz::sim::kNullEntity) {
			ignerr << "LED LINK ENTITY NOT INITIALISED" << std::endl;
		} else {

			// auto led_visuals = this->led_link.Visuals(_ecm);
			// for(gz::sim::Entity vis: led_visuals) {

			// }

			// std::vector<float> pub_data;
			ignition::msgs::Float_V pub_msg;

			// Iterate over all visual entities
			_ecm.Each<gz::sim::components::Visual, gz::sim::components::Name>(
			[&](const gz::sim::Entity &entity,
				const gz::sim::components::Visual *,
				const gz::sim::components::Name *name) -> bool
			{
				// Check if the visual is one of the LEDs we created
				if (name->Data().find("led") != std::string::npos)
				{
					std::string led_name = name->Data();
					// ignerr << "Name: " << led_name << std::endl;
					// ignerr << "SSTR: " << led_name.substr(3) << std::endl;
					int number = std::stoi(name->Data().substr(3));

					// Get the Material component
					auto materialComp = _ecm.Component<gz::sim::components::Material>(entity);

					if (materialComp)
					{
						// Get the material message
						auto materialSdf = materialComp->Data();

						float r = this->led_colours[number * 3 + 0];
						float b = this->led_colours[number * 3 + 1];
						float g = this->led_colours[number * 3 + 2];

						ignerr << "Setting r g b to " << r << " " << g << " " << b << std::endl;

						// Update the Diffuse color
						materialSdf.SetDiffuse(gz::math::Color(r, g, b, 1.0));
						materialSdf.SetEmissive(gz::math::Color(r, g, b, 1.0));  

						// Update the Material component
						_ecm.SetComponentData<gz::sim::components::Material>(entity, materialSdf);
					}

					// Lookup again to check if state was set
					materialComp = _ecm.Component<gz::sim::components::Material>(entity);
					gz::math::Color diffuse = materialComp->Data().Diffuse();
					pub_msg.add_data(diffuse.R());
					pub_msg.add_data(diffuse.G());
					pub_msg.add_data(diffuse.B());
				}
					
				return true;  // Continue iteration
			});

			// led_status_pub.Publish(pub_msg);
		}
	}

}

} // namespace gazebo
