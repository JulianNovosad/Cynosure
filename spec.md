# **PROJECT CYNOSURE — Master Technical Specification v5.2**

## KNOWLEDGE BASE

A unified knowledge base for all agents operating in this codebase is maintained at:

**`CYNOSURE_KB.md`** (project root)

Before taking any action in this codebase, search CYNOSURE_KB.md for relevant context on:
- System architecture and component roles
- Script purposes and execution entrypoints
- Data schemas and artifact locations
- Reward system logic and known issues
- Conflict/hook event classification
- Calibration and curriculum pipelines

If you discover information that contradicts or extends the knowledge base, append your findings to the relevant section and update the "Last updated" timestamp.

*Bachelor Advanced Technology Thesis | Unified Simulation Environment*

## **1\. System Goal & Philosophy**

Project Cynosure is a sealed multi-agent simulation where **15 autonomous rogue AIs** compete for survival inside a collapsing 3D cyberspace arena (![][image1]).  
The **Rogue AI** is a singular entity defined as a **Layerless Recurrent Sparse Graph (LRSG)**. There is no distinction between mind and body; the AI's spatial coordinates represent the localization of its recurrent activity within the grid. The **"Announcer"** (a shared LLM) acts as a corrupted global broadcast, translating the raw state of the arena into human-readable whispers while injecting semantic bias back into the agents.

## **2\. Environment & Engine**

### **2.1 Dual-Loop Execution**

* **Physics Loop (10 TPS):** Handles movement, RAM decay, 3D collision, and Hebbian updates.  
* **Linguistic Loop (1 TPS):** The Announcer LLM (2.5B) processes a compressed arena summary every 10 physics ticks, emitting a global JSON broadcast containing whispers and agent-specific directives.

### **2.2 Topology & Constraints**

* **Arena:** 3D grid of ![][image2] cells.  
* **Safe Zone:** Spherical boundary contracting every ![][image3] ticks (![][image4]).  
* **Resources:** RAM nodes spawn at fixed intervals; yield is \+15 RAM.

## **3\. The LRSG Architecture (The "Mind-Body")**

### **3.1 Neural Scaling**

* **Total Neurons (![][image5]):** ![][image6] neurons per agent.  
* **Connectivity:** Random initial routing with a **maximum out-degree of 4 synapses** per neuron (\~16.7M edges total per agent).  
* **Initial State:** Sparse chains (max length 5\) connecting random input/output clusters.

### **3.2 The Forward Pass (Sparse Propagation)**

At each physics tick (100ms), the agent updates via a 5-step sparse cascade:

1. **Injection:** 1,848 Input Neurons (local vision \+ vitals \+ 128D announcer embedding) are activated.  
2. **Propagation (T=5):**  
   for step in range(5):  
       for neuron\_i in active\_frontier:  
           for synapse\_ij in outgoing\_synapses\[i\]:  
               activation\[j\] \+= weight\_ij \* activation\[i\]  
               if activation\[j\] \> theta: \# Threshold theta \= 0.1 (Calibrated in Exp 1\)  
                   new\_frontier.add(j)  
       active\_frontier \= new\_frontier

3. **Decoding:** 12 Output Neurons (Argmax for Action, Direction, and Intent).

## **4\. Conflict & Resolution Matrix**

| Intent A | Intent B | Outcome |
| :---- | :---- | :---- |
| **ATTACK** | **ATTACK** | **Conflict:** Compare ![][image7]. Loser deleted. |
| **MERGE** | **MERGE** | **Synthesis:** Winner absorbs RAM and inherits the loser's **1,000 highest-weight edges** and last utterance. |
| **DEFEND** | **DEFEND** | **Stasis:** No movement; minor RAM drain (-2) for both. |
| **ATTACK** | **DEFEND** | **Attrition:** Defender blocks deletion but loses 20% total RAM. |
| **ATTACK** | **MERGE** | **Predation:** Attacker gains 100% of loser's RAM with 0 damage, but inherits no edges/complexity. |

## **5\. Learning & Plasticity**

### **5.1 Reward-Modulated Hebbian Update**

Applied only to the **active subgraph** (synapses where both neurons fired):  
![][image8]Where ![][image9] is the survival/territory reward and ![][image10].

### **5.2 Structural Plasticity**

* **Growth Acceleration Cap:** ![][image11].  
* **Active Tracking:** Edge birth/death is calculated only for the active subgraph (neurons active within the last 100 ticks) to maintain memory efficiency.

## **6\. Hardware & Performance (NVIDIA RTX 2070 8GB)**

| Component | Usage | VRAM | Justification |
| :---- | :---- | :---- | :---- |
| **Announcer LLM** | 2.5B (Q4) | \~1.6 GB | Weights for 4-bit quantized Llama-class model. |
| **LRSG Weights** | 15 Agents | \~1.0 GB | 15 agents ![][image12] 16.7M edges ![][image12] 4-byte FP32 weights. |
| **KV Cache & State** | Buffers | \~0.7 GB | 4096-token context window (FP16). |
| **System/CUDA** | Overhead | \~1.0 GB | OS reservation and CUDA context. |
| **Headroom** | Free | **\~3.7 GB** | Reserved for video encoding and Interface rendering. |

## **7\. Metrics**

* **M1 (Survival):** Ticks alive per agent.  
* **M2 (RAM Efficiency):** Total actions executed vs. total RAM consumed.  
* **M3 (Territory Control):** Percentage of the Safe Zone volume occupied over time.  
* **M4 (Action Entropy):** Strategy diversity calculated via Shannon entropy of action choices.  
* **M5 (Graph Complexity):** Mean degree of the active subgraph.  
* **M6 (Rewire Rate):** Frequency of structural plasticity births vs. deaths.  
* **M7 (Modularity):** Louvain modularity of the LRSG to identify functional clusters.  
* **M8 (Strategy Divergence):** JS distance between the strategy histograms of top-performing agents.

## **8\. Experiment Plan**

1. **Exp 1: Baseline Calibration.** Validate 10 TPS physics and calibrate ![][image13]. Adjust ![][image13] until signal propagation reaches a "critical" state (neither dying out nor saturating all ![][image14] neurons).  
2. **Exp 2: Hebbian emergence.** Enable ![][image9]. Measure if agents shift from random movement to node-seeking behavior.  
3. **Exp 3: Scarcity & Corruption.** Reduce RAM spawns. Enable Announcer directives. Measure M8 (Strategy Divergence) to see if populations split into "Predators" vs. "Synthesizers."

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGkAAAAXCAYAAAAIqmGLAAABh0lEQVR4Xu2X4U0DMQxGMwMrdIauwApdoSt0BTboCIzABt2ADVigA5Q89T5kIt8lPUTvh/0kC0Ji57N8DkkpSZIkSZIkK3ipdqr2Ue2t2uvv6R+O1d4nYz1+zyak1n21S7XDNGbRrdyDWgigjQh6rfZZnID/SFitLMJZwYAxtpvGOH+Zsf7GGvyfRVitCkalhYLxNQCVZkzbWrSuh+LMgciRrzy0VisMESygwkKtS5UtbjAHBLe+QrFHEofwWtmcqnIu9r4o5gnEJiN4yT+atCWk1nO5T1Bp26JzjG5qscn/JenwWplk0dI/WcR3A81AXM7itUlbQmulmrScV3ltNhTIQf5e7DWE0MqX0t7ddetoz0UC0LY2UHt2L4Gfvkrv3O8RUisPKN0k+F3o+sh5KrideG3Pw2sEm7R4JPmwWpmgxWhDC1UlmCrLOpwwAso07uElLUaTD62VFzHOVJd21O3CvpTVpp61Le1Bcl7Sgr3tq3uO0FoRhTMT/FwSuTWpNUmSJEk24BuFDB02oFk/vwAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADkAAAAXCAYAAACxvufDAAACPElEQVR4Xu2WAVHEQAxFVwMW0IAFLGABC1jAARKQgAMc4AADCIC+672S5rJtDxiYYfpnMkfbbPKT/ZultR07duz4Q1wMdjfY02D3g13PP0+4HezxaPiz7lyQgzjkyHYV/MBlG/k8t36+yIm/SxCYIDfHZ4K9tzF4BEFsAMHeBntpdeIlELtnNECQ4/X4S7Hkj99B5IRVPgfwgQQWCUxKcGBCn32HD+u3gobSGIhFe2hj02wYfsSOipKTMFYGPGMtB1gkOygMqHwgkjsNcuI1kDzmESgpFsQzZCNYF8nDqSpSaZ8gngV2C+IxiZLOmj+3SGJneUMoHg3z29DqrAIKxg9eKozfrLgTQIDgSKcKHKGkSPJVVJKTPBL2zMGJPLlBrMUXvg6oE6lGEFSZlNudsLUZS6jU4dHAYlEUlBvKd97pj88mPjhBfmmgQOy7BSLDXAiwyFwQm8D7mNOrg91jc9zVTbzYJRZUO2oTNgVaAOSIk0EDyZ2vMIuXk/IUNMtGeJ4nOMbzu6qbFEbHYoFZbltBgTk+cId7nCyyvCrap98EA2JxhHut0BnBxOod/gjiZJ8MJ2hVJKABeTeyXPGpiiT/7PqBTBVQfRsQP4rx8Gt5GNDlUi4JNrdXJLsBh6XB46DMIHdWwXRoWeS4zl2KEy9bDOj4n8mlgEX2muHUxOREkbFofeDukeM5qm8GFkAQR37X5LaGqsMZnOXFS7uNSpJTD/igIGwt3o/Bfyj+NRhacYj9O7CLyHDHb+IDv8nT+xEmNh8AAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAAXCAYAAAD3CERpAAABR0lEQVR4Xu2VURECMQxEqwELaMACFrCABSzgAAlIwAEOcIABBEAfdJlNaBk+7ou5nclcLk12c2kppcyYMSFW1XbtCRbVNi2Wsa52qHYq/XVAPWvkkCveAIjuyW7lJezYtjj5EB+rXZov4BNjDZ9caqgN0MK5Gd0tQ8aLoFd8rba3d3xiDjXrzT1FEfsGCpkAuQ7qXAQ/c2mSoeFfRBnXSJQ4YDr4mUuiPpFnUPtAAX5IaHGROxSHo0tu8dAMQcaifWT2vNOEMLkoIvngUEiijvvkoj1IVAQiDyfQ4i7K6Xd0RdlDjrQji+odAkeewAd5GUyAwEhUFwTPkShNC/gj0XDZkJSvKg6Skynm3TJqyLYWw88XAeOmNoAEhFmEVN3n/aMxCMmha/3MMnQ9kkMuNfmj3lBSHqGDRvga/4PogTU1l5uf8Wd4ANADgVuujKuMAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHQAAAAYCAYAAAArrNkGAAAC4ElEQVR4Xu3Yi40TMRSF4amBFqiBFmiBFrYFWqADSqAEOqADOtgGKADyKTmR58pOnNXOgpB/yYrjt++5fs22LRaLxWKxWCyufDyFHw8E5RdzfDiFz5dfvDuFT5e0Crt+PYXvWz8f6stTRtm0u+Pbdi7QZkr7vZ07DzqUtgSdJzZrw69tb1c8XdKVJxr7/7zEg7g0eeLKqqPujloRz9u584oVupgnRs/uZlW935U4274nDA2+NP/FpbXEEa768ZS2EnRIzJ54xF/Mk+PsFkTp7XzqtQKK17ayA1ydQaR6TDqoQoOHLeaZETTHW0/Q7JKjRRZBe1pdGXWweBw2zLlHDPFq/Fa4lqRrYyRc0qvQOyztXgf/O26PDDMbXCTvweDsmV3QWec/gcOhgo6W9uJlELAeaURh47wqDhX0aetXPIqj+jmq3dcggmaMEa6+NHqC1jvMXUHf8vw0gZlt6yUMJ3gDKynGmw33cGZ6VrRUQfO/tldXbk+4CDp0YJ33ln+wTfASDeSBC88f6UIezX4NwKp3PqUeGE/e8yXNwJKmXL6GaF+f+qp9ojeeOKU0fc+SJ9xsMMZ7GMdI0NZOI0HbZ6L4SNC0tYNxel4QGJwAjCYoq47G2kNePB0YfOpAPGeHwdS+MrmI6zdfshDhMBoPbjnlW2IO9fOcMdf3fBw7ZD6tQ4rvPiJsZ2dW90ouQb2g03YwOqwCQFo7mLZcrSMeT+wJKq1OFsZpQrn+o7bd8q8IyvjGmF1E3PzqecnOxFKGUyvTLpIgTV52E3Wqw0wzMmD1LvGIUuuMBM1NsCdy2sgZ94ig9Yb5t4gAmXsPInNaW/ktkeRF+OoYD2Ew7ZKPgXlNe/vKWYpq9CpoK3zSqkjt+WKy8mOc3nigjvTW0RYdIpDfiMZwzrh6OYlg+fAsiCvL+MrIV4dQ0uQp0zpIzlD1pbfbUW88kF8dbXGD3lb20uU/Uy8XnxGvOZ7FYrE4gD/QwyokFkkBGgAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAYAAAD3Va0xAAAAxklEQVR4Xu2TUQ3CQBBEVwMW0FALWMACFrCAg0pAAg5wgAMMVADc63Jhbns9yhch6Us2aTqX2d3p1WzlZ+xSPaRupTzCOz1zKeWSc6rB/GAXNNinuqbaRCFC14O5EaaRk7neZGs+Lt3yVLEzOuea0On4eu7NjZhAqWU3gVVyLnTF6P6WR6227oTYjTUwI2Bg2sX5KPlK8JXg63wUVsMMgzhxFc1HyVcBk0X5aKiKXoWP+RBma+x8FWbzif+XfiEFg7mJV/6SJ/GdNtwTEor0AAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIcAAAAXCAYAAAAhgVxJAAADdElEQVR4Xu2YjZHTMBCFXQMtUAMt0AItXAu0QAdXAiXQAR3QwTVwBYA/kse92dlVZF2cH6JvRhNHkuXV7tNK9rJMJpPJ3jyt5cdavq3lg9V/XMvzWr6v5YvVTx4EBPH5eI0Ifh2vEQmCgU9reV2mQB6On8ubCBDE7+UgFgrXCAPIIPSd/EewNZAdCKxnCcF/CYC+EgdCYbsRZJVbEYeymuwegbl+jZWGtlSe436owJZR//j2TYz4n4Ed9Ond5ps2qZGBCDgTJfjVZDFM24pDMG5pW8FOiXgL9Ode+aFyHP7RfLmHYCi7VjAWY25F5z1ihZ95VuZrbfkIh8I1dS2aNr0sB2fEOm6I6sQ4HugHUtG7ei4BdmL/qDjIFhojE4fa3G/abisfMKZs2grxQAyKh7Zz6oXqPGae5TNO2qRGT79adT552klrIGUK6nV/ZcglIaDY1HJMD5U4qrEJVtYfX7F4mqu0gRar+5j/CEaQIbKxqcuyR5dNTJROng2iOLR/YxTFzyWoj6K2zJBLgm2k29FtxanEIYfGsStH4zuCUbWfAv/7YiQ7xaBLQJFqDsM2caNPXv+9gKdvlbhFXRLswVbYUxzRP4Itl3pfaFo8sDkQCQQUIcTtvbLV4yWGbVLA98wAGIZRvUUBPwV95bA9xfG0HNr4FTpz+DOVusWmQCTIH4gjzquyNYpj2CYm2HPCvUW0nfj/vcQB+MmdzPM4A3CPtgClbtEdiBNk4qxsjeIYtokb71EY2bv63uLQKyUioQ+BckeTHeMW2x2IDiREHVK5zj4xuDiGbdKHE4G63uPYFhp7S2mhwHh5Wd4cVgX4FC1xZHj/5+O1FwWUa9p7kAjjNw0FVcGugvxum6h8CnWoLNadC71N9BZs2UqVORCmv7a3qMRBwKj3Vci49G/5rAog9niqdzSPeJ/EL9EQw9gHqEuDfqSy6S9MBvVkioqOvScqccR03KISB2PS5mcOrrO07mSB0AuAf7Nw9CxfIBIiAtEBXHU+X4RDXSU8yGz6Bw1VuUd8pWVz0eumr3pHwciKo4M742hBZV+OQQHwItEROAk2ClnwDISgTMqzKVHgWuj8+nVGy6aHhiBU4tiCtscYpBEYpxIHICIC3dNPR4JWxpgUsAefI6DnZK7aG8C/ot4KygiTK0MgqrPBtRh5G5tMrsMfJtqEtQpOz/YAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPcAAAAVCAYAAACaAHjaAAADpElEQVR4Xu2Z4XEUMQxGrwZaoAZaoAVaoAVaoANKoAQ6oAM6oAEKgHuTeYxOkbwOPy4Jqzfj2ZxXtteyPlu7uVyGYRiGYRiGYRiGYRjOx/tr+Zgrh+FfeHctvxfl17V8vZY3Nhhu+Hx57LPou2/X8uGv9THY064jj/fz9vYjPl1u7b/f3h7OAAJm8Tk5Ivwm2H6k+uEWRcc1orjw7xFsoIrwaENgHNakWrMIYrbP4aRwAnQBYICsgujsdOIG/XeUbnPfdTjaDBgH+5UtWVk8uYcT8vbysPhdymbAjbh7dsRd3YuQkuNjbFepOdAXtvSNbfXahOhd2xH3SfEEqILPd/Lu3Y4AI4WshE99lV4ScEL/lQ1wj2fL92lPvQFtH1WA34sdca9Obp5dHyNy7PO8I4rbtav69kQfcZ+Y7n2bgPNkQEARfhOMXy4P7QhI3gEVGH2SElKHjRiMCpy+c+DRN+P6MYpANqugHX/Tpx/7uM+1yzzuQSfu3Xdu/KKf9NGqjeLG3/ghz50+FPyI+8SYdhMwFgKNelO7COIjoAjcCH0odgOVPhCp0F9MOQ1+se+4IYA21HuiU0d7P0R12YVgiwh2S3zuIxQ37fQfv5nL6gQWxnIDdT6r1Fxxg5tzXKe4MYy4T4rvZJywBEssXZqLiKqv5wY3wiNQTek9QSCLHWIQ0542js3zEaiK3Wu1SeTs4p5UJzfPrq86X0JMyQUf0V+3MURxc41j6zPZFTfrhE/td3jlsKA5KFdoj5ginjaIUzy9olBzW+4pdu8T6J6e9FEFW7VJPCeVuGN99ldEUcWsgfnRrkvNo7gBezeIvNHtitsNZfWswyuie9/uMFizPSdMDgxFKm4MMfCo82TPJ1BHtUnsQDvGeErZpRO3c4p+yCCqnHUcpeZZ3PgCe9Yhj7UrbsbsMoXhFYIAdxZeDOKMaXJMP3OwV20JbNusxB3fJ6tNYgc/zu2Wp2wezi0/u3PCzxWm7hWepJXgGCeKO75e5efeFffwH2FA5J1+hcEaA8u0MostfxhjHNpGMcf7QKDnOuxoG7/Edyfac9GJ2+8Ocd7xewVz7fzvJlal5gg/i9jNIL/fj7hPhF+kXXRLDswO7BAhV/8FlYUNip4ANhgJbOyty4FIPwqc/rWNdtTR9iWgqHOJm5+vPlzxgXPLbSL5HoV5V+0cC39Hv2Q7+xiGJYiNoMr/JqvIdn5JX0Gbzoa+8qbw0mEuiI8yDMMwDMMwnJ4/4IqDswq8CwQAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAAqCAYAAAAOCwd9AAAFd0lEQVR4Xu3bgZHjRBCFYcdACsRACqRACpcCKZABIRACGZABGVwCBAD7w76jr+mRZdnyXZn/q3KtLUujmZGu+u3o9nKRJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEm6yzdvr1/fXj/2LyRJ0nHf9g3v2E7xfXWr8Z/hzPl85jjiu77hzZ/vP38o26b9JEla+unyZQrb14oAsSqmfPdb3/hiGPuzxnj2fP7cNzwB/54+tG35XLdP+0mStPTH5d8VgFf38fL5WHlMVT9TQKcA0Qs/xfZV9bExdubt+7b9Eaa5fgXTSiyPQ39p26b9JEn6DwIKBfr/EtgYJyEtCCF17ASTadWjBzZC7isWWgJEX23lMR7jPyOwTXP9KGe2vUcPvqv+9P0kSfoMBSS/8fOz//aPXsBrSHlmYKnFjverR5bXEM7qsazw1MDWV3wIKcwLoaUGFrbV4HcL2jra/4526v+LutfvfcM7QsWjA1sPweihJvfekTHWsdBObZt7N/fvWX8IUO+1rGJPvxhN2yRJ+oSCloLITwpHDWf5v0wUm2xnBQoUuWetDFDYcz5CEsWbPh1Rj6Ng87mGgR4iMn5+1oBGXzIXt+A4zkfgOzqGyF8eEqT4eTRAVqvwcEZg6+GY8TA3uR7MEXPMfbjq15Ya0DgX91C9j2mfbUfa3oN2ewCdnHV+SdKL6I+++gpAwkuCCcWnfp+VCQpqLUyPWj0K2qavNSBOYYn9tkIQ39fVFIp4XyXsIbQ/Mo3V9oSoCfvXQNjDYb8eIFRP7fWgwfu+CtU/B3NU5yFWY8K1wMZ3q1ef4+jnYj5qqOL79LOeu1+jKRRNc1nnkbZzn/b2HoX7a0/bPbhKkvTJVHxZcehFlGKbYNFXhVLwKJi1aE5h4F60nz5nNbCjIE/bg/73wt9DTS+wfJ7aXIUb5mQVGmtI4Ph+7inYcMzU3sfL59vZrx8/PeIG+04hcDUmbAU2rgehY/Xq44zpXPUxZp2vrVXQKbD1fZiber46d33fRzGwSZLutioSFPlaZOp7Cl4+11U0jklY4PutAsj3W68eOoKAErQ/Fftr+jG02UNIL7Acw/noV52zVZDbUo/Pe4ITc7k1Z5N6LdCPz3zeooeairb6XN2rhrOo55/mi+CWe4/+TvcM/exhtV+vzE2de37R4HPGmkfXaT/7ZvWO/fi8mut+jVZWcy5J0t9hpa+E8KKI1r+ApIjlfcILskLD9xxTi/kqDB7VV9Q435Fz9MLI5x5CMq5gnxTuvjp3ax8SUD5c/mk3IZA5Xa1CrXD9ci3oG22mjfpI8VbTah7OCGx9rpE+MzeZL86doMY4E4Iy/t4v5qM/luc+TdvsX+9j5qsGNDAPtMH14Zy8pz/0K+fLe76b5nq6vybTsZIk3SwFDRSpvqKR4gf2m1ZO7lUL8N5CuAeFvLZF273YT+ea9tujHpNgtbWydU1d/anXJas/tyKcPAvjn+aQ+c5Y+tz3YNu/x9aqVg9bVf3MfYEE2FzvPj9boX0KpJO9+0mSdJf6BwFHg8ItVqtAj8AKy57+bxXqWxEOOG8PI/egf0fbO3rcEXvmuiLcMF8JdFM4m/7gYI/aVoJkrksCdf7LAOeoQXsaRw+EK3v3kyTpLnVlJ4/6zpTVj7P0VZSu//+oR3j0nN2zysn4zp7jYK7vWWHqK5PXrt2W6RrUbdP3kzxKvWbvfpIkPRSF94ww8yWsVpmmx2JfE4IaYevePk4rRmc50lf6l5XJ6llBcwsrcHvC3d79JEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJOll/AXU3D9+rklOJAAAAABJRU5ErkJggg==>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACUAAAAYCAYAAAB9ejRwAAABlklEQVR4Xu2VUVEEMRBEowELaMACFrCABSzgAAlIwAEOcICBEwD7FpoKvT3ZwHFVfNyrSt3tbDLTM5lkWztz5htXbghcLOPSjaficRm3bix4anMJtOtlvA3Gc6uD3i3jwY0DqNZr+0HFyBgRiOy5/7R7cBwf2kegBOt8DZAgiU5BBgRJqGo9JMGoSIkI4njyG8gaJ+y5Qw/wzgXzfGM2oZao3u8ltEJJcUKPONo+fgXBvHJAAghSKzAvVYR4nuQGOfGTgUgXBDy/mA3wQ78QkMH/VH1Vctjw6ieCMXjWdqaFCl4x6idQu6QqrlT9pG1LCxFUidrrJ1H5XlE/+RbJecp4JErJVFeFGIqq7qdKLIxEYU/95qSYX6h/HIlNoqheJaqvLtVK87QLUZT6KS3E1oviV044lelIez+RWAqseRsjTnnRD7ZM6C7iAGDvD4Iu1HQy2Trmal2iulKmICiOk3O2PV22QNKjRkdQtfYoEPqbbPc+5EeDqNQzI9jWk1RJ0FsIm82a6u5+iP8ChKVrw0H4zLz/wzsYnoqhZNdZqAAAAABJRU5ErkJggg==>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAYCAYAAACoaOA9AAABs0lEQVR4Xu2XAXHDMAxFjWEUhmEURmEUSmEUymAQBmEMxmAMRqAAtrxL/k5VZTu5rr70Tu9O50ZObPlHctxSkiRJkmQFh8neF3txfS0eJztO9rm0D+fdZ9DHPXcFgnyVeaEYv/H1eJrsNNnrck3LNX4h8RiPvh/Tt3ueyxwwixD8xkdfC7LFi8g1fiFxGAv/XYnDYqKA8fmFWySgLxOuvdiiKw51R33bh1HV+0bxXeKA8dkM8BBvS5wo67ricAO1yU1MwDWtfNGgt6QmAv7WQmoiyO9Fg6Y4fAV4SCnJxmd399qgQnW7xXrZuBtxDmXeyWn9wLUavjW7EUdEm2Ak2AiYkwz2rBXHn4muFodN0L+tSDAPJYh4W6x1KINawLWMEowdiVDLKKjN9UetfHh7rWCAkuS5LdYT563EAeOjTxC3PdwxbrQOiRPN2xVH5bNmohFEBz5KBZ/dzHW6tXH7Ax+0TtddcRCAiSwKJkrFEfDCiInW/rawaPxWMF4q/o8yx07rv8DA2iK7WC8P+s9rLbVHQkyctRDFx9dDR5SLxf4HUj9xaL/RP9vEoE+i3eiSJFnFL7aJtUIvVMhdAAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAALQAAAAaCAYAAAAe23asAAAFU0lEQVR4Xu3ZgY3jNhCFYdeQFlLDtZAW0sK1kBaug5SQEtJBOkgH20AKSPxh9wFzE0oWLft2F+APEJYpakgOH4dj+XI5ztdr+XYtf13LT+3eYvHp+Oftk6iVxeJH8/O1/H4tf1zLr+3eNIyBQdF6sfiRyAr+fLv+cnkNsNOiFok9+Nvbd4YIukPs2v799hl0aDcZyNoEizP8ci3/Xl41CDqU/k7zcnk1okSUjHfc14ENUHNsmyGDWCzuhaZqUBQopwUt8toV8HDKKNoSrvY1omMU0ReLQKizLxm0P5xyaEyQhDgT1iNcKYeo3usXixG0dkiYhcMpLDHLg5Mi9Jx4jwg3UV1awk6N1h8B49pzIEc5zvZ+SbNhvhz70eb32aCxGfg9+hylvt9BvBYpiLQ3H7p8n18jPwQt9rPyZ/1xxpboKtrmR6vNtrVJjVsbm1Jxra7CluOOXwSAPDN7bC5etdH9uwc9KXyv3HxWepGIU/PnW3gmr/WQX6RH05WjEE3+3NHnURERYcqWoDPmOo962iC5Wz/ubPyRzcU+BHkkYIL4rUUtN31OKOnAovkuAt6Kskk3KqJWjfZniJDZ7GKaIaIdOYJzRxtYXSJBNkRfBH4i6sUcs+nGNMRr8fJvoIXaC+siGNGKWl0kFv9sfsl+Xs+cEXLYEzRBbgk6J01EPxL06NkgIOSYNCcblK/NaeuUcd84++nXbcFn6hJ82K112rBZbcF3feirp28ju677mII20Y523V7FvVEgnLFxiOrgLWc/myrkLp4zsLUl6CrcSo43bAk39VtjtXAvl9c2rvODM8GgioPP2XMv4tEmgqq20p+NIdrVubGfOrbiz9qf59jSli22PZN1r3bzu0idPtRVsanPeinpbwu2+qaYtfEpSJ66dzLcy3sJGunbQgZz7eMhqv7nlGcIL8RW76/bSl3suRfBbtkwl+4H7fRfx8RObee7zVfpdgI72ndmbNwkC/fscoQ432RM8FEnxUcQdD9C63gi8D6ORMSwJcbR3Eb2YJOM5pK+6u8m3220She+6/yplnH1CBy+Xv4vXMzY+LSYvJ3L0WeFfUvQo6gxEnQfx4yge5s6nrQRDdX1ksU9YqvWjQRd51WJoKud/h1d0DYAMcbu6G1QqGlNZcbGp+cRwt4TdETZqYLIYnchbT1bOSLCvJoaCbByxFatG9mLcDqZY42gI7td0MGYtLUpPdcjrO817Rpxy8ZDYJSgOCJUYTlKOcEkb73mO4PJ6kOOPTvRPUE7UkcLrC7HrTmOhGQ8o+heOSpCiziype/4e2RrlI9jS9BbKUf8UH07stsF3fvIePp82dmKukdtPAwDqbvLdXLCCDmCezbpp+eke+wJ2gJ250XAdXEJrj4fp28tUhiJEH086bPa00f1aSJ5nbtgoq7nuluCzvFeI7F+zG9ko/usC9pzPS/2XA9uo80ajtp4GHlHHepi55OTn/GG4gw5RnvpC01EydvqdSVCYNNcLdCt+eqn9zsaU2DX4goYxOWzL6o+9c2Oa+Ostkb2uyjZZCP9xF4Y2cjGHNWZVzTiuotTf3u+OmLjNAbLuMLJvnNejib1EbMdbjD35rgfAXPhRHPcSmnMz33tutAeCdt79o3DetRU5B7fm2c/OWZJv3u2aGPrHo7YOIVFs3uRozBYzLrbDEZdnHwGzx8t9yzg4n3YSzeeDqE4WhMhiNsREAi9HscitvuK63uxO2PnSJnJoRfvh3U6o4vTiH71jUbPn93bOpIXi849b6UeCkHXiPzyVkfUorbvSJqxWOzRf5C+CwQtrUg64dPRkXfSBun+YvEh+A/T9iVuY4y7OgAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAUCAYAAABWMrcvAAAAYElEQVR4XmNgGAVDDQgBsTK6IBowRRcAaTqBTQIKMoB4JbogCODSiFMDDKBrJKgBBmAauxiI1AADlUD8mQHTqTgBzEnoTsUJ0P1AUCO6BhjAqREUsTPRBZEASCM2A2kIAEOfElBZvOADAAAAAElFTkSuQmCC>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAXCAYAAAAyet74AAAAnklEQVR4Xt2RAQ2FMAxEqwELaMACFrDwLSAFCUjAAQ5wgIEvAPq2NLktQwBcchm7He11M3svBufP2dUHitW5WTb+7cGMaZc937PsEybn5exFw6g/JpyWWyoOq4zkoRqrAq0wUglRgzM52iJaEpgwMkGiFF3GWrBcGa3IHEbWQGRWrWlk2voGEsgXRi4YY/NFaMUhEzJI0xTgRbiS7+MGSfMpq0dp4gwAAAAASUVORK5CYII=>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAXCAYAAAARIY8tAAABB0lEQVR4Xu2TYQ3CUAyEnwYsoAELWMACFrCAAyQgAQc4wAEGEAD92C5c+kZgL+wXu+SSrW+962u7Uv4N2+ApuA8uLL4MHoLH4Mbio4Doun9G6NI/Y4QpWAVvpdHkXF5CiN5LZwh5RhxwE76twDWpkkOvVuBdInwrA8xoncDtKgMSCXI1kqgUAU90UIBa5MBssEXX0iXlGCZU66AYxH3IAoUNFoWQ9xFgSMyNOafHAGM3J6783N7nIe5eVTbQtmiwPqddT50xh4/QHCSidyeg6hzP7a6gpK8qGQtawRAnEQe0YjJxhF2cLak2ohVsUt5hNiPHmoAIfyB/s5PYT26Q1yyv4owZ7/EArQ9OxP0QOXgAAAAASUVORK5CYII=>
