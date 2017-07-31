
orders_prior=pd.merge(orders,prior,on="order_id",how="inner")
orders_prior.sort(by=['user_id','order_num'])

class features():
    def __init__(self,user_id):
        self.user_id=user_id
        self.product_dict={}
        self.days_since_list=[]
        self.order_number=0
    def update(self,order_id,user_id,eval_set,order_number,order_dow,hour_of_day,days_since,product_id,cart_order,reordered):
        if hour_of_day < 8 :
            period_of_day="mid_night"
        elif hour_of_day < 13:
            period_of_day="morning"
        elif hour_of_day < 19:
            period_of_day="afternoon"
        else :
            period_of_day="night"
        if self.order_number != order_number:
            if order_number ==1:
                self.days_since_list.append(0)
            else:
                self.days_since_list.append(days_since)
            self.order_number=order_number
        if not self.product_dict.has_key(product_id):
            self.product_dict[product_id]={"period":set(),"dow":list(),"cart":list(),"order_nbr":list()}
        self.product_dict[product_id]["period"].add(period_of_day)
        self.product_dict[product_id]["dow"].append(order_dow)
        self.product_dict[product_id]["cart"].append(cart_order)
        self.product_dict[product_id]["order_nbr"].append(order_number)
    def output(self):
        output_list=[]
        for product_id,values in self.product_dict.iteritems():
            period=list(values["period"])
            order_nbr=values["order_nbr"]
            order_diff=[]
            nbr_p=None
            for nbr in order_nbr:
                if nbr_p is None:
                    nbr_p=nbr
                    continue
                else:
                    order_diff.append(sum(self.days_since_list[nbr_p:nbr]))
                    nbr_p=nbr
            order_diff.append(sum(self.days_since_list[nbr:]))
            output_list.append([self.user_id,product_id,period,order_nbr,order_diff])
        return output_list

i=0
user_prod_features=[]
user_id_p=None
order_num_p=None
for rec in orders_prior.values:
    order_id,user_id,eval_set,order_number,order_dow,hour_of_day,days_since,product_id,cart_order,reordered=rec
    if user_id != 199176:
        continue
    i+=1
    if user_id!=user_id_p:
        if user_id_p is not None:
            user_prod_features.extend(user_feature.output())
        user_feature=features(user_id)
        user_id_p=user_id
    user_feature.update(order_id,user_id,eval_set,order_number,order_dow,hour_of_day,days_since,product_id,cart_order,reordered)
user_prod_features.extend(user_feature.output())
