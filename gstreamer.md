Gstreamer Plugins
=====================

### Class definition

#### struct
This holds the data for the element

#### class (struct)
element class name, base class name

#### GstElementDetails
* element details in user readable form
* details stored in Gstreamer XML
* contains :
	- short name
	- tree of the plugin
	- detailed description
	- contact info
* set via ```gst_element_class_set_details```

#### GstStaticPadTemplate
* description of a pad
* contains:
	- short name
	- direction (GST_PAD_SINK / _SRC)
	- availability (GST_PAD_ALWAYS / _SOMETIMES / _REQUEST)
	- supported capabilities (ANY, etc.)
* set via ``` ```

### Init Functions

#### _base_init()
* initialize class and base class properties
* called on each new class creation

#### _class_init()
* initialize class once, setting up its:
	- signals
	- arguments
	- virtual functions

#### _init()
* initialize specific instance of this plugin

### Properties and Caps

#### _set_property()
* when declared arguments / properties are to be set

#### _get_property()
* when an argument / property value is to be fetched

#### _setcaps()
* called during caps negotiation
* response is yes (TRUE) / no (FALSE)

### State Change

#### _change_state()
* called when state of the plugin needs to be changed
* states include : NULL / READY / PAUSED / PLAYING

### Data processing

#### _chain()
* all data processing happens here
* called when new data buffer becomes available at the sink pad

### Scheduling / Modes
Plugin manybe running in one of the modes:

#### MASTER MODE / TASK RUNNER MODE
* these run the pipeline (e.g. - ALSA src)

#### PULL  MODE
* control data flow in the pipeline, may allow random access (e.g. - filesrc)

#### PUSH MODE
* tuned to process data when sink pad gets populated (e.g. - volume control)
