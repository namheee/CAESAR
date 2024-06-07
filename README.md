# CAESAR
Canalizing kernel for cell fate determination

+ We used the pyboolnet (https://github.com/hklarner/pyboolnet) to import bnet file.
+ Some example networks are provided in the `network` directory.
  
### Python:
```python
networkName = 'mESC_Network_2019'
undesired = '0010011010001001'
# {'CH': '0', 'Esrrb': '0', 'Gbx2': '1', 'Klf2': '0', 'Klf4': '0', 'LIF': '1', 'MEKERK': '1', 'Nanog': '0', 'Oct4': '1', 'PD': '0', 'Sall4': '0', 'Sox2': '0', 'Stat3': '1', 'Tbx3': '0', 'Tcf3': '0', 'Tfcp2l1': '1'}
desired = '0111111110111101'
# {'CH': '0', 'Esrrb': '1', 'Gbx2': '1', 'Klf2': '1', 'Klf4': '1', 'LIF': '1', 'MEKERK': '1', 'Nanog': '1', 'Oct4': '1', 'PD': '0', 'Sall4': '1', 'Sox2': '1', 'Stat3': '1', 'Tbx3': '1', 'Tcf3': '0', 'Tfcp2l1': '1'}

from CAESAR.mainCAESAR import caesar
desired_pstate, alg, runtime, node_fbl_info = caesar(networkName, undesired, desired, fbl_thres=10)
```
