---
tags:
  - spiking neural networks
  - e-prop
  - NEST
  - DVS Gesture
  - MC_Maze
---

+++ {"part": "abstract"}
This tutorial presents native NEST e-prop for recurrent spiking neural networks using two spike-based classification tasks: DVS Gesture event-camera data and MC_Maze neural population activity. The workflow connects input preprocessing, recurrent SNN training, diagnostic learning curves, and interactive visualization of learned synaptic-weight changes.
+++

# Native NEST e-prop for Spiking Neural Networks

## From synaptic plasticity to trainable spiking models

The brain learns by changing the strength of synaptic connections between neurons. This ability, known as synaptic plasticity, allows neural circuits to adapt their activity based on experience. Rather than communicating with continuous values, biological neurons exchange information through brief electrical events called spikes. Learning therefore emerges from the interaction between spike timing, neural dynamics, and changes in synaptic strength.

Spiking neural networks (SNNs) are computational models inspired by this principle. In an SNN, neurons do not transmit continuous activations at every step, as in standard artificial neural networks. Instead, they integrate incoming signals over time and emit a spike only when their internal state reaches a threshold [@Gerstner2002; @Neftci2019]. This event-driven mode of computation makes SNNs especially well suited for temporal and sparse data, such as event-camera recordings or neural spike trains. It also makes them attractive for neuromorphic computing and for modeling biological neural circuits more faithfully.

A common neuron model used in SNNs is the leaky integrate-and-fire (LIF) neuron. A LIF neuron can be understood as a simple dynamical system with three main ingredients:

- it integrates incoming synaptic inputs over time;
- its membrane potential leaks back toward rest in the absence of input;
- and it fires a spike when the membrane potential crosses a threshold.

After emitting a spike, the neuron is reset and begins integrating again. In practice, this gives rise to a characteristic behavior: the membrane potential gradually rises with input, decays when input is absent and produces a discrete spike when it exceeds threshold. In native NEST e-prop models, recurrent neurons are implemented with the `eprop_iaf` model, which is based on a leaky integrate-and-fire neuron and is designed to support eligibility-propagation learning.

:::{figure} static/fig1_lif.png
:label: fig-lif
:align: center

Basic behavior of a leaky integrate-and-fire neuron. Incoming signals are integrated over time, the membrane potential decays in the absence of input, and a spike is emitted when threshold is crossed.
:::

Training recurrent SNNs is challenging. In conventional recurrent neural networks, learning is often performed with backpropagation through time (BPTT). BPTT works by unfolding the recurrent network across all time steps, storing the relevant states, and propagating errors backward through the full temporal history. This approach is powerful, but it is computationally heavy and biologically difficult to interpret, because synaptic updates depend on non-local information propagated backward in time.

Eligibility propagation (e-prop) provides a more biologically inspired alternative for recurrent spiking networks [@Bellec2020]. The key idea is to factor learning into two parts:

- a local eligibility trace at each synapse, which captures the recent interaction between pre- and post-synaptic activity;
- and a learning signal, which provides information about whether the network output should be increased or corrected.

In simplified form:

$$
\text{weight update} \approx \text{eligibility trace} \times \text{learning signal}.
$$

This decomposition is important because the eligibility trace can be computed locally at each synapse during the forward simulation, without storing the full network history for backpropagation. The learning signal then modulates these traces to produce weight updates. In this sense, e-prop preserves temporal learning while remaining closer to biologically plausible synaptic plasticity than BPTT.

:::{figure} static/fig2_bptt_vs_eprop.png
:label: fig-bptt-eprop
:align: center

Conceptual comparison between backpropagation through time (BPTT) and eligibility propagation (e-prop). BPTT propagates errors backward through the full temporal history, whereas e-prop combines local eligibility traces with learning signals.
:::

In this tutorial, we use NEST Simulator, a simulation framework for spiking neural networks [@Gewaltig2007]. In NEST, a model is built by defining neuron and device populations, connecting them with synapses, and then running a simulation over time.

:::{figure} static/fig3_nest_workflow.png
:label: fig-nest-workflow
:align: center

Basic workflow in NEST Simulator. A model is first built by defining neuron populations and external/input devices, then connecting them with weighted synapses. The network dynamics are simulated over time, and the resulting activity can be recorded through devices such as spike recorders or multimeters, producing outputs such as spike rasters, firing rates, or membrane-potential traces.
:::

The examples in this tutorial focus on native NEST e-prop applied to two spike-based classification tasks. The first is DVS Gesture, in which event-camera recordings are converted into spike-based inputs for gesture recognition. The second is MC_Maze, in which neural population activity is used to classify movement-related conditions. Together, these two examples illustrate how the same e-prop learning framework can be applied to very different kinds of temporal data.

# Datasets and input representations

The two examples used in this tutorial are based on different types of temporal data. The first one, DVS Gesture, comes from an event-based camera and represents visual motion through asynchronous events. The second one, MC_Maze, comes from neural population recordings during a reaching task and already consists of biological spike trains.

Although the datasets have different origins, both are converted into the same general type of input for the model: a spike raster, where each input channel emits spikes over time. This makes them suitable for recurrent spiking neural networks trained with native NEST e-prop.

## Dataset 1: DVS Gesture

DVS Gesture is recorded with an event-based camera [@Amir2017]. Unlike a standard camera, which records full image frames at fixed intervals, a DVS camera records only changes in brightness. Each event contains a spatial position, a timestamp, and a polarity: $(x, y, t, \mathrm{polarity})$.

The polarity indicates the direction of the brightness change:

- ON event = brightness increased
- OFF event = brightness decreased

This representation is naturally sparse and temporal, which makes it well suited for spiking neural networks.

For the tutorial run, a simplified configuration is used. Four gesture labels are selected, with 60 training samples and 20 test samples per label. Each event stream is converted into a compact spike-based representation before being sent to the NEST network.

:::{figure} static/fig4_dvs_preprocessing.png
:label: fig-dvs-preprocessing
:align: center

DVS Gesture input representation and preprocessing. (a) Example event-camera sample visualized as accumulated OFF and ON events. (b) Events are spatially downsampled from a $128 \times 128 \times 2$ resolution to a $16 \times 16 \times 2$ representation, corresponding to 512 input channels. (c) The event stream is temporally normalized to 500 ms and binned with a 5 ms time resolution. (d) The final model input is a 500 ms spike raster with 512 input channels.
:::

This compressed representation is a practical tutorial choice. It keeps the simulation lightweight and easier to reproduce, while preserving the main event-based structure of the gesture, but it is not meant to be the only possible or optimal representation of the DVS Gesture dataset.

## Dataset 2: MC_Maze

MC_Maze contains neural population activity recorded during a reaching task [@Pei2021; @Churchland2022]. In contrast to DVS Gesture, the input is already neural and spike-based: each channel corresponds to a recorded unit.

The task used here is an 8-class classification problem based on the maze or movement condition. For each trial, spike times are extracted from the 182 recorded neural units and converted into a 1500 ms input sequence aligned to movement onset before being sent to NEST.

:::{figure} static/fig5_mc_maze_input.png
:label: fig-mc-maze-input
:align: center

MC_Maze task context and neural input representation for native NEST e-prop. (a) An example reaching trial provides behavioral context, including the hand trajectory, cursor trajectory, and target location. Task timing signals are used to align trials and define the analysis window used for classification. (b) Neural population activity is represented as a spike raster across the 182 recorded units. (c) The final model input is a 1500 ms spike raster sent to NEST. The behavioral trajectory and velocity plots provide task context, whereas the neural raster is the actual input used by the model.
:::

For the MC_Maze model, the readout is divided into three temporal segments of 500 ms. This segmentation is used at the output level to preserve coarse temporal information across the trial. It does not change the input format itself: the network still receives one continuous 1500 ms spike sequence. This segmentation is not the only possible choice, and alternative readout designs could be tested.

# Native NEST e-prop network architecture

Both models use the same general native NEST e-prop structure. Preprocessed spike trains are first provided to the network as input spike sources, then relayed to a recurrent spiking population, and finally projected to task-specific readout neurons.

The recurrent module contains 150 leaky integrate-and-fire (LIF) neurons implemented with the native NEST `eprop_iaf` model [@NESTeprop]. These neurons form the recurrent core of the network. Their activity is shaped by three trainable weight matrices:

- `W_in_rec`: input relay layer $\rightarrow$ recurrent neurons
- `W_rec_rec`: recurrent neurons $\rightarrow$ recurrent neurons
- `W_rec_out`: recurrent neurons $\rightarrow$ readout neurons

These matrices are sparse: not every possible connection is present in the network. As a result, each neuron receives input from only a subset of neurons, and the exact connectivity pattern depends on the chosen sparsity parameters.

The input stage converts the preprocessed spike rasters into NEST spike sources. These sources are connected one-to-one to an input relay layer, implemented with parrot neurons. The relay layer then projects to the recurrent population through sparse plastic synapses. Inside the recurrent module, neurons are also connected to one another through sparse recurrent plastic synapses. Finally, the recurrent population projects to the readout layer through trainable recurrent-to-readout connections.

The recurrent-to-recurrent matrix, `W_rec_rec`, is particularly important because it defines how activity circulates within the recurrent population. In other words, it controls the internal temporal dynamics of the network, rather than simply passing information forward from input to output.

During training, the readout activity is compared with target teaching signals. The resulting error information is sent back toward the recurrent population through an `eprop_learning_signal_connection`, represented by the feedback signal $B_{jk}$. Synaptic weights are then updated through `eprop_synapse` connections using the combination of local eligibility traces and the learning signal [@NESTeprop].

During evaluation, the learning rate is set to zero. The same network is simulated, but the synaptic weights are no longer updated, so the model only produces predictions.

:::{figure} static/fig6_dvs_architecture.png
:label: fig-dvs-architecture
:align: center

Native NEST e-prop architecture for DVS Gesture classification. The DVS Gesture model receives a compressed 500 ms spike raster derived from the preprocessed event-camera recording, with 512 input channels ($16 \times 16 \times 2$) corresponding to the spatially downsampled ON and OFF event streams. These input spikes are provided to NEST through spike generators and relayed by parrot neurons before reaching a population of 150 recurrent LIF neurons. Plastic synapses connect the relay layer to the recurrent population, recurrent neurons to each other, and recurrent neurons to 4 readout neurons, one for each gesture class. During training, readout activity is compared with class-specific target signals, and the resulting e-prop learning signal modulates synaptic updates together with local eligibility traces.
:::

:::{figure} static/fig7_mc_architecture.png
:label: fig-mc-architecture
:align: center

Native NEST e-prop architecture for MC_Maze classification. The MC_Maze model receives a 1500 ms spike raster extracted from 182 recorded neural units. Input spike trains are delivered through spike generators and relayed by parrot neurons to a recurrent population of 150 LIF neurons. Plastic synapses are used for the input-to-recurrent, recurrent-to-recurrent, and recurrent-to-readout connections. The readout layer contains 24 neurons, organized as 8 classes repeated across 3 temporal readout segments. These segments provide class-specific activity over successive portions of the trial, and the final class prediction is obtained by combining readout activity across the three segments. During training, the readout activity is compared with class-specific target signals, and the error is returned to the recurrent population through the e-prop feedback signal $B_{jk}$.
:::

# Implementation overview

The full commented scripts are provided separately as downloadable files. Both scripts follow the same general workflow: load the dataset, convert samples into spike trains, build the native NEST e-prop network, train with plastic `eprop_synapse` connections, evaluate with learning disabled, and save the resulting metrics, figures, predictions, and weight matrices.

The main difference between the two implementations is the input and target structure. DVS Gesture uses compressed event-camera spike rasters with 4 class readouts, while MC_Maze uses neural spike rasters and a temporal readout organized into 3 segments $\times$ 8 classes.

The two complete, commented training scripts used for the tutorial are available directly with the article, together with the scripts used to generate the interactive diagnostic curves and the standalone 3D network browsers:

- {download}`DVS Gesture native NEST e-prop training script <downloads/dvs_gesture_native_nest_eprop.py>`
- {download}`MC_Maze native NEST e-prop training script <downloads/mc_maze_native_nest_eprop.py>`
- {download}`Interactive training-curve generator <downloads/make_interactive_training_figures.py>`
- {download}`3D network-browser generator <downloads/make_3d_unified_network_browser_per_family.py>`

# Training results and diagnostics

During training, three quantities were monitored: classification accuracy, loss, and recurrent firing rate. Accuracy and loss summarize task performance, while the recurrent firing rate is used as an additional diagnostic specific to spiking networks.

The firing rate is particularly useful because an SNN can fail even when the training loop runs correctly. If recurrent activity is too low, the network may be almost silent and unable to propagate information. If activity is too high, the recurrent dynamics may become unstable or overly driven. A useful run should therefore show learning progress while keeping recurrent activity in a reasonable range.

:::{attention} Interactive training curves
Use the metric selector to switch between accuracy, MSE-like loss, and mean recurrent firing rate. Hover over any point to inspect its exact iteration and value. If a NeuroLibre/Evidence preview shows only the static placeholder, attach the computational runtime and execute the corresponding figure cell.
:::

## DVS Gesture

:::{figure} #fig-dvs-results-cell
:label: fig-dvs-results
:placeholder: ./static/fig8_dvs_results_placeholder.png

DVS Gesture training results. The selector switches between classification accuracy, MSE-like loss, and mean recurrent firing rate. Training, train-evaluation, and test phases are displayed as separate traces, and hovering over a point reports its exact values.
:::

For DVS Gesture, the training accuracy increases rapidly during the first training iterations and reaches approximately 0.8–0.9 near the end of training. When learning is disabled, train-evaluation and test accuracy remain high, indicating that the network learned the four-class gesture task and generalized reasonably well to held-out samples.

The loss decreases steadily across training, showing that the readout activity becomes progressively closer to the target signal. At the same time, the recurrent firing rate decreases from a relatively high initial value to a lower and stable regime around the end of training. This suggests that learning does not simply increase network activity but instead modifies the recurrent dynamics while keeping the network active.

A standalone HTML version is also available: {download}`DVS Gesture interactive training curves <downloads/dvs_gesture_interactive_training_curves.html>`.

## MC_Maze

:::{figure} #fig-mc-results-cell
:label: fig-mc-results
:placeholder: ./static/fig9_mc_results_placeholder.png

MC_Maze training results. The selector switches between classification accuracy, MSE-like loss, and mean recurrent firing rate. Training, train-evaluation, and validation phases are displayed as separate traces, and hovering over a point reports its exact values.
:::

For MC_Maze, the training accuracy increases over time and reaches high values near the end of training. Train-evaluation accuracy remains high, whereas validation accuracy is more variable and lower, suggesting that the model learns structure from the neural spike trains but generalizes less stably than in the DVS Gesture example.

The loss decreases throughout training, which confirms that the readout signals move closer to the target signals. However, the validation loss remains noisier near the end, consistent with the more variable validation accuracy. The recurrent firing rate stays active throughout the run, mostly around the mid-20 Hz range, indicating that the network is not silent and that the recurrent dynamics remain stable.

Overall, the MC_Maze run should be interpreted as a working native NEST e-prop proof of concept on neural population data rather than as an optimized decoding model.

A standalone HTML version is also available: {download}`MC_Maze interactive training curves <downloads/mc_maze_interactive_training_curves.html>`.

# Visualizing learned weight changes

The trained models save the initial and final weight matrices. This makes it possible to inspect how learning changed the network.

The 3D visualization displays neurons as nodes and non-zero synaptic connections as edges. It can show `W_in_rec`, `W_rec_rec`, and `W_rec_out` for three different states: `initial`, `final`, and `delta = final - initial`.

In the delta view:

- blue edges = weights decreased during learning;
- red edges = weights increased during learning;
- edge thickness = magnitude of the change.

Zero-weight connections are not displayed. This keeps the visualization readable and reflects the sparse structure of the network.

The recurrent-to-recurrent matrix, `W_rec_rec`, is usually the most informative view because it shows how the recurrent core of the model changes during training. The browser can also display input-to-recurrent and recurrent-to-readout connections separately or together. The top-k sliders control how many of the strongest non-zero connections are displayed.

::::{seealso} DVS Gesture interactive 3D network browser
:class: dropdown

:::{iframe} https://eprop-wired-dashboard.onrender.com/network/dvs_gesture
:width: 100%
:height: 760px
:border: 0
:::

::::

::::{seealso} MC_Maze interactive 3D network browser
:class: dropdown

:::{iframe} https://eprop-wired-dashboard.onrender.com/network/mc_maze
:width: 100%
:height: 760px
:border: 0
:::

::::

The standalone HTML files can also be downloaded and opened locally:

- {download}`DVS Gesture 3D network browser <downloads/dvs_gesture_3d_network.html>`
- {download}`MC_Maze 3D network browser <downloads/mc_maze_3d_network.html>`

**Figure 10 — Unified 3D network visualization for DVS Gesture and MC_Maze.** Input, recurrent, and readout neurons can be displayed together. Separate top-k sliders control the number of visible `W_in_rec`, `W_rec_rec`, and `W_rec_out` connections.

This visualization should be interpreted as an exploratory view of the learned network. A single edge should not be treated as a complete explanation of a class decision, but the global pattern of weight changes provides useful insight into how the recurrent model was modified during training.

# Parameters worth changing

The scripts can be modified to test different network and training configurations. The most relevant parameters are those that affect the recurrent dynamics, the learning rule, and the input representation:

| Parameter | Meaning |
| --- | --- |
| `N_REC` | number of recurrent neurons |
| `N_ITER_TRAIN` | number of training iterations |
| `ETA_TRAIN` | learning rate |
| `GROUP_SIZE` | number of samples per training group |
| `TARGET_AMPLITUDE` | target readout amplitude |
| `W_IN_SCALE` | input weight scale |
| `W_REC_SCALE` | recurrent weight scale |
| `W_OUT_SCALE` | readout weight scale |
| `FEEDBACK_SCALE` | learning-signal feedback scale |

For DVS Gesture, additional preprocessing parameters include:

- number of labels;
- samples per label;
- spatial resolution;
- sequence duration;
- temporal bin size;
- compressed vs full-time representation.

For MC_Maze, useful variants include:

- number of temporal readout segments;
- sequence duration;
- alignment event;
- number of classes.

These parameters affect both classification performance and spiking dynamics. For example, increasing the input or background drive can make the network more active, but too much activity may reduce stability. Conversely, weights that are too small can make the recurrent population nearly silent. The recurrent firing rate curve is therefore useful when tuning the model, because it helps distinguish a network that fails to learn from one that simply does not spike enough.

# Limitations and possible extensions

The examples presented here use simplified training runs. The parameters were selected to make the simulations easier to understand and reproduce, not to exhaustively optimize performance.

The DVS Gesture model uses a compressed 500 ms representation. This makes the tutorial faster, but it does not preserve the full temporal structure of the original event stream.

The MC_Maze model uses three fixed 500 ms temporal readout segments. This provides a simple early/middle/late temporal structure, but other readout strategies could be tested, such as one global 1500 ms readout, more segments, or overlapping windows.

The 3D visualization is useful for inspecting learned weight changes, but it does not provide a complete mechanistic explanation of the model. Individual neurons and connections should not be over-interpreted without additional analyses.

Finally, e-prop is more biologically inspired than BPTT, but the model remains a simplified computational system. The readout targets, learning rates, and training setup are engineered components.

# Conclusion

Native NEST e-prop provides a practical framework for training recurrent spiking neural networks with eligibility traces and learning signals.

In this tutorial, the same learning principle is applied to two different spike-based inputs: event-camera data from DVS Gesture and neural population activity from MC_Maze. The resulting workflow connects data preprocessing, recurrent SNN training, diagnostic curves, and interactive weight visualization.

Overall, the tutorial shows how the learning process can be followed from the input spike representation to the recurrent network dynamics, the readout activity, and the resulting synaptic weight changes. By combining diagnostic curves with interactive weight visualization, the examples provide a practical way to inspect how recurrent spiking networks learn from temporal data.

+++ {"part": "data_availability"}
The training logs used to build the interactive diagnostic curves are included with the article as downloadable CSV files. The standalone interactive HTML visualizations and their generator scripts are also included. The original DVS Gesture and MC_Maze datasets remain referenced through their respective source publications and repositories.
+++

# Supporting files

- {download}`DVS Gesture training log <downloads/dvs_training_log.csv>`
- {download}`MC_Maze training log <downloads/mc_maze_training_log.csv>`
- {download}`DVS Gesture interactive results HTML <downloads/dvs_gesture_interactive_training_curves.html>`
- {download}`MC_Maze interactive results HTML <downloads/mc_maze_interactive_training_curves.html>`
- {download}`DVS Gesture 3D network HTML <downloads/dvs_gesture_3d_network.html>`
- {download}`MC_Maze 3D network HTML <downloads/mc_maze_3d_network.html>`
- {download}`DVS Gesture native NEST e-prop training script <downloads/dvs_gesture_native_nest_eprop.py>`
- {download}`MC_Maze native NEST e-prop training script <downloads/mc_maze_native_nest_eprop.py>`
- {download}`Interactive training-curve generator <downloads/make_interactive_training_figures.py>`
- {download}`3D network-browser generator <downloads/make_3d_unified_network_browser_per_family.py>`
