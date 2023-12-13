from entropy_lens.models.explainer import Explainer
from entropy_lens.logic.metrics import formula_consistency
from torch.utils.data import DataLoader, TensorDataset, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
import time
import numpy as np
import copy


# the data is for each client
def local_train(user_id, epochs, train_loader, val_loader, test_loader, n_classes, n_concepts, concept_names, base_dir,
                results_list, explanations, model, topk_explanations, max_minterm_complexity: int = None,
                      max_accuracy: bool = False, x_to_bool: int = 0.5,
                      y_to_one_hot: bool = False, verbose: bool = False, logic_generation_threshold=0.7):
    checkpoint_callback = ModelCheckpoint(dirpath=base_dir, monitor='val_loss', save_top_k=1)
    trainer = Trainer(max_epochs=epochs, gpus=1, auto_lr_find=True, deterministic=True,
                      check_val_every_n_epoch=1, default_root_dir=base_dir,
                      weights_save_path=base_dir, callbacks=[checkpoint_callback],
                      enable_progress_bar=False, enable_model_summary=False)

    start = time.time()
    trainer.fit(model, train_loader, val_loader)
    # print(f"Concept mask: {model.model[0].concept_mask}")
    model.freeze()

    model_results = trainer.test(model, dataloaders=test_loader)
    if model_results[0]['test_acc_epoch'] > logic_generation_threshold:
        for j in range(n_classes):
            n_used_concepts = sum(model.model[0].concept_mask[j] > 0.5)
            # print(f"Extracted concepts: {n_used_concepts}")
        results, f = model.explain_class(train_loader, val_loader, test_loader, topk_explanations=topk_explanations,
                                         max_accuracy=max_accuracy, concept_names=concept_names,
                                         max_minterm_complexity=max_minterm_complexity,
                                         x_to_bool=x_to_bool, y_to_one_hot=y_to_one_hot, verbose=verbose)
        end = time.time() - start
        results['model_accuracy'] = model_results[0]['test_acc_epoch']
        results['extraction_time'] = end

        results_list.append(results)
        extracted_concepts = []
        all_concepts = model.model[0].concept_mask[0] > 0.5
        common_concepts = model.model[0].concept_mask[0] > 0.5
        for j in range(n_classes):
            n_used_concepts = sum(model.model[0].concept_mask[j] > 0.5)
            # print(f"Extracted concepts: {n_used_concepts}")
            # print(f"Explanation: {f[j]['explanation']}")
            # print(f"Explanation accuracy: {f[j]['explanation_accuracy']}")
            # explanations[j].append(f[j]['explanation'])
            extracted_concepts.append(n_used_concepts)
            all_concepts += model.model[0].concept_mask[j] > 0.5
            common_concepts *= model.model[0].concept_mask[j] > 0.5

        results['extracted_concepts'] = np.mean(extracted_concepts)
        results['common_concepts_ratio'] = sum(common_concepts) / sum(all_concepts)

        return copy.deepcopy(model.state_dict()), model.model[0].concept_mask, results, f
    else:
        return copy.deepcopy(model.state_dict()), model.model[0].concept_mask, None, None