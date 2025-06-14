import Parameters
import measure_AD_Nm
import measure_AD_Ns
import measure_RGlpsNm
import measure_RGlpsNs

'''环境更新的测量数据'''
# RGlps_Measure = measure_RGlpsNm.measure_Nm(para=Parameters.measure_data())
# if RGlps_Measure.measure_mode_init == 'True':
#     RGlps_Measure.start_measure()
# while RGlps_Measure.measure_num < RGlps_Measure.para['measure_num_tot']:
#     RGlps_Measure.start_measure()

# RGlps_Sample = measure_RGlpsNs.Sampling_Ns(para=Parameters.sample_data())
# if RGlps_Sample.sample_mode_init == 'True':
#     RGlps_Sample.start_sample(ns_init=RGlps_Sample.Ns_init)
# while RGlps_Sample.sample_num < RGlps_Sample.para['sample_num_tot']:
#     RGlps_Sample.start_sample(ns_init=RGlps_Sample.Ns_init)

'''自动微分的测量数据'''
# RGlps_Measure_AD = measure_AD_Nm.measure_Nm_AD(para=Parameters.measure_data())
# if RGlps_Measure_AD.measure_mode_init == 'True':
#     RGlps_Measure_AD.start_measure()
# while RGlps_Measure_AD.measure_num < RGlps_Measure_AD.para['measure_num_tot']:
#     RGlps_Measure_AD.start_measure()

# import argparse
# args = argparse.ArgumentParser(description='Arguments Parser Of Sample')
# args.add_argument('--')

RGlps_Sample_AD = measure_AD_Ns.Sampling_Ns_AD(para=Parameters.sample_data())
if RGlps_Sample_AD.sample_mode_init == 'True':
    RGlps_Sample_AD.start_sample()
while RGlps_Sample_AD.sample_num < RGlps_Sample_AD.para['sample_num_tot']:
    RGlps_Sample_AD.start_sample()
