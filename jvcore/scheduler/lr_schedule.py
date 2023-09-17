class CustomLR(object):
    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.optimizer = optimizer
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.base_lr = optimizer.lr
        self.base_lr_pg = [pg.get("lr") for pg in optimizer.param_groups]

    def get_lr(self, base_lr, now_lr):
        ## Get the current lr
        if self.last_epoch == 0:
            return base_lr
        return base_lr * self.gamma ** self.last_epoch


    def step(self):
        ## Update the lr, External interface function
        self.last_epoch += 1
        self.update_lr()

            
    def update_lr(self):
        # How to update the lr
        self.optimizer.lr = self.get_lr(self.base_lr, self.optimizer.lr)
        for i, param_group in enumerate(self.optimizer.param_groups):
            if param_group.get("lr") != None:
                param_group["lr"] = self.get_lr(self.base_lr_pg[i], param_group["lr"])