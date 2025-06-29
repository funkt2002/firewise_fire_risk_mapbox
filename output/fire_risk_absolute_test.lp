\* FireRisk_Absolute_test *\
Maximize
MaximizeTotalScore: 3.3251388667 weight_hagri_fb + 1.5823884171 weight_hbrn
 + 1.5069160451 weight_hvhsz + 3.13727686791 weight_hwui
 + 2.49209895686 weight_neigh1d + 1.20406038522 weight_par_buf_sl
 + 2.96583563469 weight_qtrmi
Subject To
WeightSum: weight_hagri_fb + weight_hbrn + weight_hvhsz + weight_hwui
 + weight_neigh1d + weight_par_buf_sl + weight_qtrmi = 1
Bounds
 weight_hagri_fb <= 1
 weight_hbrn <= 1
 weight_hvhsz <= 1
 weight_hwui <= 1
 weight_neigh1d <= 1
 weight_par_buf_sl <= 1
 weight_qtrmi <= 1
End
