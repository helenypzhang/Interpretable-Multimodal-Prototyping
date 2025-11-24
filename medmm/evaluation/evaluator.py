import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, classification_report
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, concordance_index_ipcw
from sksurv.util import Surv

from .build import EVALUATOR_REGISTRY

def survival_AUC(df1, df2, event_times, risk):
    # import pdb;pdb.set_trace()
    
    event_times = np.array(event_times)
    times = np.percentile(event_times, np.linspace(10, 81, 15))
    
    surv1 = np.array(df1, dtype=int)
    risk2 = np.array(risk)
    surv2 = np.array(df2, dtype=int)

    surv1 = np.core.records.fromarrays(surv1[:, [1,0]].transpose(), names='obs, survival_months', formats = '?, i8')
    surv2 = np.core.records.fromarrays(surv2[:, [1,0]].transpose(), names='obs, survival_months', formats = '?, i8')
    _, iauc = cumulative_dynamic_auc(surv1, surv2, risk2, times)
    return iauc

def auc_com(y_true, y_pred, num_cls):
    # y_true 是真实标签
    # test_predictions 是模型的预测概率
    # import pdb;pdb.set_trace()

    auc_scores = []
    for class_idx in range(num_cls):
        y_true_class = [1 if y == class_idx else 0 for y in y_true]
        y_pred_class = [pred[class_idx] for pred in y_pred]
        
        auc = roc_auc_score(y_true_class, y_pred_class)
        auc_scores.append(auc)
    auc_value = sum(auc_scores)/len(auc_scores)*100
    # import pdb;pdb.set_trace()
    return auc_value

class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
    
@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self._m_out = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]

        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]
        
        self._m_out.extend([mo.squeeze().cpu().numpy()])
        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        cls_report = classification_report(self._y_true, 
                        self._y_pred, 
                        labels=np.unique(self._y_true),
                        zero_division=1)
        
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )   
        auc_score = auc_com(self._y_true, np.array(self._m_out), max(np.unique(self._y_true)+1))

        
        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["AUC"] = auc_score
        results["macro_f1"] = macro_f1
        # results["cls_report"] = cls_report

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* AUC: {auc_score:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%\n"
        )

        print(cls_report)

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )
            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results


@EVALUATOR_REGISTRY.register()
class Classification_UMEML(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self._m_out = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]

        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]
        
        self._m_out.extend([mo.squeeze().cpu().numpy()])
        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        cls_report = classification_report(self._y_true, 
                        self._y_pred, 
                        labels=np.unique(self._y_true),
                        zero_division=1)
        
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )   
        if len(self._m_out[0].shape) == 1:
            auc_score = auc_com(self._y_true, np.array(self._m_out), max(np.unique(self._y_true)+1))
        else:
            auc_score = auc_com(self._y_true, np.concatenate(self._m_out, axis=0), max(np.unique(self._y_true)+1))

        
        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["AUC"] = auc_score
        results["macro_f1"] = macro_f1
        # results["cls_report"] = cls_report

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* AUC: {auc_score:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%\n"
        )

        print(cls_report)

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )
            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results

@EVALUATOR_REGISTRY.register()
class Survival(EvaluatorBase):
    """Evaluator for survival."""

    def __init__(self, cfg,  **kwargs):
        super().__init__(cfg)
        self._total = 0
        self._all_risk_scores = []
        self._all_censorships = []
        self._all_event_times = []


    def reset(self):
        self._total = 0
        self._all_risk_scores = []
        self._all_censorships = []
        self._all_event_times = []


    def process(self, S, censorship, survival_month):
        self._total += censorship.shape[0]
        risk = -torch.sum(S, dim=1).cpu().numpy()
        self._all_risk_scores.extend(risk)
        self._all_censorships.extend(censorship.cpu().numpy())
        self._all_event_times.extend(survival_month.cpu().numpy())




    def evaluate(self):
        results = OrderedDict()
        c_index = 100.0 * concordance_index_censored((1-np.array(self._all_censorships)).astype(bool),np.array(self._all_event_times), np.array(self._all_risk_scores), tied_tol=1e-08)[0]
        # i_auc = 100.0 * survival_AUC(train_data, test_data, self._all_event_times, self._all_risk_scores)
        
        # The first value will be returned by trainer.test()
        results["c_index"] = c_index
        # results["I-AUC"] = i_auc


        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* C-Index: {c_index:.1f}%\n"
            # f"* I-AUC: {i_auc:.1f}%\n"

        )


        return results

@EVALUATOR_REGISTRY.register()
class Survival_UMEML(EvaluatorBase):
    """Evaluator for survival."""

    def __init__(self, cfg,  all_survival=None,  bins=None, **kwargs):
        super().__init__(cfg)
        self.all_survival = all_survival
        self.bins = bins
        
        self._total = 0
        self._all_risk_scores = []
        self._all_censorships = []
        self._all_event_times = []
        self._all_patient_ids = []
        self.all_risk_by_bin_scores = []


    def reset(self):
        self._total = 0
        self._all_risk_scores = []
        self._all_censorships = []
        self._all_event_times = []
        self._all_patient_ids = []
        self.all_risk_by_bin_scores = []


    def process(self, patient_id, logits, censorship, survival_month):
        self._total += censorship.shape[0]
        hazards = torch.sigmoid(logits)
    

        S = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(S, dim=1).cpu().numpy()
        # print(risk)
        
        self._all_risk_scores.extend(risk)
        self._all_patient_ids.extend(patient_id)
        self._all_censorships.extend(censorship.cpu().numpy())
        self._all_event_times.extend(survival_month.cpu().numpy())
        self.all_risk_by_bin_scores.extend(S.cpu().numpy())


        
    def evaluate(self):
  
        results = OrderedDict()
        all_risk_scores = np.delete(self._all_risk_scores, np.argwhere(np.isnan(self._all_risk_scores)))
        all_censorships = np.delete(self._all_censorships, np.argwhere(np.isnan(self._all_risk_scores)))
        all_event_times = np.delete(self._all_event_times, np.argwhere(np.isnan(self._all_risk_scores)))
        
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

        c_index_ipcw = 0.

        # change the datatype of survival test to calculate metrics 
        try:
            survival_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
        except:
            print("Problem converting survival test datatype, so all metrics 0.")
            return c_index, c_index_ipcw
        # cindex2 (cindex_ipcw)
        try:
            c_index_ipcw = concordance_index_ipcw(self.all_survival, survival_test, estimate=all_risk_scores)[0]
        except:
            print('An error occured while computing c-index ipcw')
            c_index_ipcw = 0.
        
 
        
        c_index, c_index_ipcw = 100.0 * c_index, 100.0 * c_index_ipcw

        results["c_index"] = c_index
        results["c_index_ipcw"] = c_index_ipcw



        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* cindex: {c_index:.2f}%\n"
            f"* cindex_ipcw: {c_index_ipcw:.2f}%\n"

        )


        return results
