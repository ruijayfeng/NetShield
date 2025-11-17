graph [
  directed 0
  multigraph 0
  node [
    id 0
    label "Node_0"
    type "router"
    cpu_cores 4
    memory_gb 16
    location "Beijing"
  ]
  node [
    id 1
    label "Node_1"
    type "switch"
    cpu_cores 2
    memory_gb 8
    location "Shanghai"
  ]
  node [
    id 2
    label "Node_2"
    type "server"
    cpu_cores 8
    memory_gb 32
    location "Guangzhou"
  ]
  node [
    id 3
    label "Node_3"
    type "switch"
    cpu_cores 2
    memory_gb 8
    location "Shenzhen"
  ]
  node [
    id 4
    label "Node_4"
    type "client"
    cpu_cores 2
    memory_gb 4
    location "Hangzhou"
  ]
  node [
    id 5
    label "Node_5"
    type "server"
    cpu_cores 12
    memory_gb 64
    location "Nanjing"
  ]
  node [
    id 6
    label "Node_6"
    type "client"
    cpu_cores 4
    memory_gb 8
    location "Wuhan"
  ]
  node [
    id 7
    label "Node_7"
    type "router"
    cpu_cores 6
    memory_gb 24
    location "Chengdu"
  ]
  edge [
    source 0
    target 1
    weight 0.85
    bandwidth_mbps 1000
    distance_km 1200
    link_type "fiber"
  ]
  edge [
    source 0
    target 3
    weight 0.92
    bandwidth_mbps 500
    distance_km 2800
    link_type "satellite"
  ]
  edge [
    source 1
    target 2
    weight 0.88
    bandwidth_mbps 1000
    distance_km 800
    link_type "fiber"
  ]
  edge [
    source 1
    target 4
    weight 0.76
    bandwidth_mbps 100
    distance_km 450
    link_type "wireless"
  ]
  edge [
    source 2
    target 3
    weight 0.94
    bandwidth_mbps 1000
    distance_km 120
    link_type "fiber"
  ]
  edge [
    source 2
    target 5
    weight 0.91
    bandwidth_mbps 1000
    distance_km 1500
    link_type "fiber"
  ]
  edge [
    source 3
    target 6
    weight 0.73
    bandwidth_mbps 100
    distance_km 950
    link_type "wireless"
  ]
  edge [
    source 4
    target 6
    weight 0.68
    bandwidth_mbps 50
    distance_km 680
    link_type "wireless"
  ]
  edge [
    source 5
    target 7
    weight 0.87
    bandwidth_mbps 500
    distance_km 1800
    link_type "fiber"
  ]
  edge [
    source 6
    target 7
    weight 0.79
    bandwidth_mbps 100
    distance_km 1100
    link_type "wireless"
  ]
]