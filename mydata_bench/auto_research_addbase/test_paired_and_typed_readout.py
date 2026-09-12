import unittest
import numpy as np
import torch
from .paired_analysis import paired,holm
from mydata_bench.meter_eval.readout import known_logit_progress_trajectory,official_progress_trajectory,canonicalize

class TypedReadoutTests(unittest.TestCase):
 def test_known_logits_are_not_probabilities_when_sum_is_one(self):
  logits=[-2.,-1.,0.,0.,0.,0.,0.,0.,0.,4.]
  self.assertEqual(sum(logits),1.)
  expected=float((torch.tensor(logits).softmax(-1)*torch.linspace(0.,1.,10)).sum())
  actual=known_logit_progress_trajectory([logits])[0]
  self.assertEqual(actual,expected);self.assertTrue(0<actual<1)
  self.assertNotEqual(actual,official_progress_trajectory([logits])[0])
  row={'status':'ok','progress':actual,'native_progress_logits':[logits],'score_readout_type':'softmax_known_logits'}
  self.assertEqual(canonicalize(row)['progress'],actual)

class ClusterTests(unittest.TestCase):
 def fixture(self):
  labels={};baseline={};candidate={}
  for i,(group,label,bp,cp) in enumerate([('a',5,3,5),('a',1,3,2),('b',5,5,5),('b',1,2,1),('b',1,4,2),('b',1,2,1)]):
   key=str(i);labels[key]={'video_sha256':group,'reward':label,'subset':'task'}
   baseline[key]={'status':'ok','progress':(bp-1)/4,'native_prediction':bp}
   candidate[key]={'status':'ok','progress':(cp-1)/4,'native_prediction':cp}
  return labels,baseline,candidate
 def test_cluster_bootstrap_invariant_to_uniform_duplication(self):
  labels,baseline,candidate=self.fixture();first=paired(candidate,baseline,labels,labels,draws=1000)
  for values in [labels,baseline,candidate]:values.update({key+'_duplicate':dict(value) for key,value in list(values.items())})
  second=paired(candidate,baseline,labels,labels,draws=1000)
  self.assertEqual(first['n_video_clusters'],2);self.assertEqual(second['n_video_clusters'],2)
  for name in ['mae','accuracy','suc_accuracy','fail_accuracy']:
   self.assertEqual(first['effects'][name]['delta_candidate_minus_baseline'],second['effects'][name]['delta_candidate_minus_baseline'])
   self.assertEqual(first['effects'][name]['ci95_video_cluster_percentile'],second['effects'][name]['ci95_video_cluster_percentile'])
 def test_zero_effect_and_incomplete_gate(self):
  labels,baseline,candidate=self.fixture();zero=paired(baseline,baseline,labels,labels,draws=1000)
  self.assertEqual(zero['effects']['mae']['ci95_video_cluster_percentile'],[0.,0.])
  self.assertEqual(zero['joint_direction_intersection_union_p'],1.)
  del candidate['0'];result=paired(candidate,baseline,labels,labels,draws=1000)
  self.assertFalse(result['complete']);self.assertFalse(result['point_estimate_10pp_gate']);self.assertEqual(result['n_paired'],5)
 def test_holm_monotonic(self):self.assertEqual(holm({'a':.01,'b':.02,'c':.4}),{'a':.03,'b':.04,'c':.4})
if __name__=='__main__':unittest.main()
