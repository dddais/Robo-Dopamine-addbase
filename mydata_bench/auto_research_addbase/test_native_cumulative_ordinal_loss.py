import pytest
import torch
from mydata_bench.auto_research_addbase.native_cumulative_ordinal_loss import cumulative_log_loss,native_grade_target,native_cumulative_ordinal_loss


def test_binary_reduction_and_strict_propriety_with_all_middle_grades():
    logits=torch.tensor([.2,-.4],requires_grad=True);q=torch.tensor([.3,.7])
    assert torch.allclose(cumulative_log_loss(logits,q),-(q*torch.log_softmax(logits,0)).sum(),atol=1e-7)
    q=torch.tensor([.05,.15,.2,.25,.35]);optimal=q.log().requires_grad_(True)
    expected=sum(q[i]*native_cumulative_ordinal_loss(optimal,i+1,'qwen') for i in range(5))
    loss=cumulative_log_loss(optimal,q)
    assert torch.allclose(loss,expected,atol=1e-7)
    grad,=torch.autograd.grad(loss,optimal)
    assert torch.max(torch.abs(grad))<1e-6
    perturbed=optimal.detach()+torch.tensor([.3,-.2,.4,-.1,.1])
    assert cumulative_log_loss(perturbed,q)>loss
    # Equal probability of the true middle grade, but distant mistakes cost more.
    near=torch.tensor([.01,.4,.18,.4,.01]).log()
    far=torch.tensor([.4,.01,.18,.01,.4]).log()
    assert native_cumulative_ordinal_loss(far,3,'rr')>native_cumulative_ordinal_loss(near,3,'rr')


def test_native_targets_extreme_logits_and_finite_difference_gradient():
    for model,size in [('qwen',5),('rr',5),('meter',10)]:
        for grade in range(1,6):
            target=native_grade_target(torch.zeros(size),grade,model)
            assert torch.allclose(target.sum(),torch.tensor(1.))
            assert torch.allclose((target*torch.linspace(0,1,size)).sum(),torch.tensor((grade-1)/4),atol=1e-7)
            if model=='meter' and grade in [2,3,4]:assert torch.count_nonzero(target)==2
        extreme=torch.linspace(-1000,1000,size).requires_grad_(True)
        loss=native_cumulative_ordinal_loss(extreme,1,model)
        gradient,=torch.autograd.grad(loss,extreme)
        assert torch.isfinite(loss) and torch.isfinite(gradient).all()
        assert gradient[-1]>0 and gradient[0]<0
    logits=torch.tensor([.2,-.3,.1,.4,-.2],requires_grad=True)
    gradient,=torch.autograd.grad(native_cumulative_ordinal_loss(logits,3,'qwen'),logits)
    numeric=[];eps=1e-3
    for i in range(5):
        plus=logits.detach().clone();minus=plus.clone();plus[i]+=eps;minus[i]-=eps
        numeric.append((native_cumulative_ordinal_loss(plus,3,'qwen')-native_cumulative_ordinal_loss(minus,3,'qwen'))/(2*eps))
    assert torch.allclose(gradient,torch.stack(numeric),atol=4e-5,rtol=2e-3)
    with pytest.raises(ValueError,match='grade1'):
        native_grade_target(torch.zeros(5),True,'qwen')
