#!/usr/bin/env python3
"""Train/test isolated natural expression seeds for the 100-tool dataset.

This module intentionally does not import or reuse the old hard phrase list.
The seed text avoids exact catalog triggers: tool_id, name_zh, keyword_label,
and positive_phrases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_CATALOG_PATH = Path(__file__).resolve().parents[2] / "configs" / "tool_catalog.jsonl"


_TRAIN_TEMPLATES = (
    "帮我{0}",
    "我想{0}",
    "麻烦先{0}",
    "能不能{0}",
)

_TEST_TEMPLATES = (
    "请帮我{0}",
    "我需要{0}",
)


_SEEDS: dict[str, dict[str, list[str]]] = {
    "get_order_list": {"train": ["把最近买过的东西按时间整理出来", "看看这周下过哪些单子", "找出前几天购买记录里的编号", "列出账号里还在处理的交易"], "test": ["翻一下昨天到今天的购买记录", "确认这个账号近一个月有哪些交易"]},
    "get_order_detail": {"train": ["看某笔购买里包含哪些商品和金额", "核对这单的收件信息和优惠", "打开那笔交易里的售后入口", "确认这次购买的每项费用"], "test": ["把指定编号对应的内容展开给我", "核实那单里地址和折扣是否正确"]},
    "get_order_payment_status": {"train": ["确认那笔钱有没有扣成功", "看看收银台失败的原因", "核对付款渠道和截止时间", "判断这单还需不需要继续付钱"], "test": ["查清这次付款到底完成没有", "看一下未完成付款还能保留多久"]},
    "get_order_invoice": {"train": ["看看这单能不能开报销凭证", "找一下电子税务票据下载入口", "核对抬头和税号是否填对", "确认能否重新出具报销材料"], "test": ["帮我拿到这笔消费的报销凭证", "检查公司抬头票据有没有生成"]},
    "modify_order_address": {"train": ["把还没发出的包裹改到新住址", "替这单更换收件人电话", "确认现在能否换收货地点", "把配送信息调整成公司地址"], "test": ["把这单送到另一个小区", "看看出库前还能不能改收件资料"]},
    "cancel_order": {"train": ["把这笔还没处理完的购买撤掉", "看看当前阶段能不能不买了", "提交终止这单的请求", "给我结束刚下的那笔交易"], "test": ["这单我不要了帮我处理", "确认是否还能撤销这次购买"]},
    "get_bill_summary": {"train": ["算清这单商品费运送费和减免", "核对周期消费里每项扣款", "看看实付由哪些部分组成", "整理退款抵扣后的费用明细"], "test": ["把这笔消费的各项金额拆开", "确认优惠和运送费怎么计入总价"]},
    "get_payment_methods": {"train": ["看看这单还能用哪些付款渠道", "列出账号可选的收银方式和限额", "确认哪种付款有额外减免", "比较当前可用的银行卡和余额渠道"], "test": ["这单现在可以通过什么途径付", "看一下每种收银渠道的额度限制"]},
    "create_payment_retry": {"train": ["给失败的付款重新生成入口", "让待付那单再弹一次收银台", "重新拉起这笔交易的付款链接", "帮我继续完成刚才没付成的钱"], "test": ["把未完成的付款再试一次", "给这单补一个新的收银入口"]},
    "verify_payment_risk": {"train": ["查清为什么付款被拦住", "核对是否有重复扣费异常", "看看账号付款被限制的原因", "判断这笔高额支付有没有被拦截"], "test": ["确认收银失败是不是安全策略导致", "排查这次扣款异常的风控原因"]},
    "get_installment_plan": {"train": ["算这单分几次还更合适", "看看每期要付多少钱和手续费", "确认这件商品是否支持按月付款", "列出可选期数及对应成本"], "test": ["帮我评估按期还款的选择", "看这单拆成多期后每月多少"]},
    "get_order_coupon_usage": {"train": ["看看这单用了哪张抵扣券", "核对优惠门槛和实际减掉的钱", "确认退货后那张券怎么处理", "找出订单里券类减免的来源"], "test": ["帮我弄清这笔交易的券抵扣", "查看已用权益的退回规则"]},
    "get_order_item_snapshot": {"train": ["看下单当时页面上的标题和规格", "核对购买瞬间的售价和图片", "找回当时商品承诺的服务内容", "确认成交时选的是哪个版本"], "test": ["还原这单付款时的商品页面信息", "检查购买那刻的规格与价格"]},
    "confirm_order_receipt": {"train": ["帮这单标记已经拿到了", "把已送达包裹设为完成", "提交收妥商品的动作", "结束这笔已收到货的流程"], "test": ["我已经拿到东西了请处理", "把签收后的交易状态确认掉"]},
    "get_order_afterpay_status": {"train": ["看看先拿货后还款的截止日", "确认尾款还剩多少没结", "查这单账期还款有没有逾期", "核对补款金额和最后期限"], "test": ["帮我看先享后还的欠款情况", "确认这笔账期消费何时要结清"]},
    "get_item_info": {"train": ["看看这个链接里的基础介绍", "打开商品标题品牌和主图信息", "了解这件东西的卖点和服务", "确认页面上的标价和基础权益"], "test": ["帮我概览这个商品页面", "看一下这件货的品牌和核心信息"]},
    "get_item_specs": {"train": ["看看材质尺寸功率这些规格", "核对型号容量和成分说明", "确认这个版本的硬件参数", "找出页面里的详细规格表"], "test": ["帮我看这件东西的尺寸材质", "查清该型号的容量和功率"]},
    "get_sku_stock": {"train": ["看看黑色大码还有没有货", "确认我所在地区能不能买这个版本", "查某个颜色尺码是否还能下单", "看仓里可卖数量和补货情况"], "test": ["帮我确认这个规格还有没有现货", "查看本地能发的版本是否有货"]},
    "get_price_history": {"train": ["看看这件东西最近涨跌过没有", "查一查过去一段时间的最低价", "确认现在是不是近期便宜点", "拉一下这个商品的价格曲线"], "test": ["帮我判断现在买贵不贵", "看近几周售价有没有波动"]},
    "compare_item_price": {"train": ["把几个店的到手金额放一起比", "看看同款在哪边更划算", "比较不同平台含运送费后的价格", "核对两个商品服务差异和总花费"], "test": ["帮我挑出更省钱的购买渠道", "对比这几款最后实际要付多少"]},
    "get_item_reviews": {"train": ["翻一下买家反馈和晒图", "看看评分低的人主要说什么", "找这个商品的追评内容", "查看近期用户使用后的评价"], "test": ["帮我看真实买家怎么说", "查一下带图反馈里有没有问题"]},
    "summarize_item_reviews": {"train": ["把买家反馈归纳成优缺点", "总结质量和尺码方面的共性", "提炼评价里适合的人群", "归纳差评集中抱怨的点"], "test": ["帮我快速概括用户口碑", "把评论里的风险点总结出来"]},
    "get_item_qa": {"train": ["找买家问过的问题和商家回答", "看看关于安装的问答有没有提到", "查询别人购买前的常见疑问", "翻这个商品下面的咨询回复"], "test": ["帮我看大家问过哪些细节", "查一下页面问答里有没有答案"]},
    "check_item_authenticity": {"train": ["确认这件货有没有授权证明", "看看平台验真服务怎么说明", "核对防伪渠道和资质材料", "判断这家卖的是否有正规保障"], "test": ["帮我查这款是不是正规来源", "看一下商品有没有验真说明"]},
    "get_item_warranty": {"train": ["看看这件货能保多久", "确认哪些故障能免费处理", "找官方维修入口和延长服务", "核对售后保障范围和期限"], "test": ["确认质保年限", "查一下坏了以后哪里处理"]},
    "get_size_recommendation": {"train": ["按我的身高体重建议穿多大", "结合尺码表帮我选合适号码", "看看这双鞋我该买哪个码", "根据过往试穿反馈给个尺寸建议"], "test": ["帮我判断这件衣服选哪号", "按我的体型推荐合身规格"]},
    "get_similar_items": {"train": ["找几件风格接近的款", "按这个商品推荐同类型选择", "给我一些预算内的相近款", "根据浏览偏好找替代风格"], "test": ["帮我看看有没有差不多的商品", "推荐几款同风格但不同店的"]},
    "get_replacement_items": {"train": ["目标没货时找能替换的选择", "预算超了帮我换成更合适的", "这款不适合就推荐别的可买款", "找满足同样用途的备选商品"], "test": ["帮我换一款能买的备选", "这件不行的话找用途相同的"]},
    "get_brand_store_info": {"train": ["看看这个牌子的官方店信息", "核对店铺评分和发货地", "查这家店的资质和服务承诺", "确认售后由哪个店铺负责"], "test": ["帮我了解这个品牌店靠不靠谱", "查看店铺资质和评分"]},
    "get_category_rank": {"train": ["看看这个类目最近卖得好的款", "找同品类里口碑靠前的商品", "查看新品或热卖排行", "按趋势筛出当前热门选择"], "test": ["帮我看这个分类的热门款", "查一下同类商品里评价靠前的"]},
    "get_new_arrivals": {"train": ["看看这个牌子最近上了什么", "查类目里刚发布的商品", "找最近几天新推出的款", "确认预售的新款有哪些"], "test": ["帮我看近期刚上的选择", "查这个分类的新发布商品"]},
    "get_low_stock_alert": {"train": ["没货时来货了提醒我", "这款补上后给我发通知", "把低余量商品加入提醒", "有可买数量时通知账号"], "test": ["帮我设置补上货再通知", "这件恢复可买时提醒我"]},
    "check_presale_status": {"train": ["看看预订商品什么时候补尾款", "确认定金规则和发出时间", "查这款提前售卖的资格限制", "核对尾款开始和结束时间"], "test": ["帮我看预约购买的发出安排", "确认定金商品后续怎么付"]},
    "get_current_promotions": {"train": ["看看现在有哪些满减和直降", "查当前店铺正在参加的促销", "找这个类目正在进行的秒抢优惠", "确认平台主题会场可用权益"], "test": ["看看今天有什么优惠场", "查这家店目前的促销安排"]},
    "get_available_coupons": {"train": ["看看我能领哪些抵扣券", "查购物车里可用的券", "找这个商品能叠的权益券", "列出账号当前可领取的减免券"], "test": ["帮我找能省钱的券", "查看这单可以用哪些券"]},
    "apply_coupon": {"train": ["把那张减免券用到购物车", "给这单套上最合适的券", "使用选中的抵扣权益并看结果", "把可选券应用到当前结算"], "test": ["把券抵到这次购买里", "用这张权益看看能少付多少"]},
    "calc_discount": {"train": ["算活动和券叠完最后多少钱", "预估会员折扣后的到手金额", "把满减积分都算进总价", "计算购物车最终应付"], "test": ["帮我算这次实际要花多少", "看所有优惠叠加后便宜多少"]},
    "get_member_level": {"train": ["看看我的会员现在处在哪一级", "查成长值和保级还差多少", "确认当前等级有哪些权益", "核对账号会员进度"], "test": ["帮我看会员身份和权益", "查我离下一级还差多少"]},
    "get_member_price": {"train": ["看看会员身份下这件多少钱", "确认专属价能不能和活动同用", "查当前等级购买这款的价格", "核对会员限购和专享折扣"], "test": ["帮我看会员买这款便宜多少", "确认专属价格的使用限制"]},
    "calc_points_deduction": {"train": ["算积分最多能抵多少钱", "看看抵扣后还要付多少", "确认积分使用门槛和比例", "把可用积分折成金额"], "test": ["帮我算积分能省多少", "看用积分后应付金额"]},
    "get_points_balance": {"train": ["看看账号还有多少积分可用", "查即将过期的积分数量", "核对冻结和可用积分", "翻一下最近积分变动"], "test": ["帮我看积分还剩多少", "查看快过期的积分"]},
    "query_price_protection": {"train": ["看看买贵了能不能退差额", "确认这单是否符合降价补偿", "查可申请差价的截止日", "判断商品降价后是否能补钱"], "test": ["帮我看这单能不能补差", "确认降价补偿还在有效期吗"]},
    "create_price_protection_claim": {"train": ["帮符合条件的单提交补差", "发起降价后退差额的请求", "把这笔买贵的订单送去审核", "申请把差价返还到账户"], "test": ["帮我提交降价补偿", "把这单的差额退款申请上去"]},
    "build_bundle_plan": {"train": ["差一点满减时推荐凑的商品", "按门槛帮我补齐购物车金额", "找几件小东西让总价达标", "根据偏好生成凑满方案"], "test": ["帮我凑到优惠门槛", "找合适搭配让满减生效"]},
    "get_flash_sale_status": {"train": ["看看抢购什么时候开始", "确认限时低价还剩多少名额", "查秒点开售的余量和限制", "看当前能不能参与限时抢"], "test": ["帮我看限时抢购进度", "确认这场开抢时间和剩余量"]},
    "reserve_promotion_slot": {"train": ["帮我预约开抢提醒", "把会员日资格先占上", "预订限时场次的参与资格", "设置活动开始前通知"], "test": ["帮我订一个促销提醒", "提前锁定这场活动资格"]},
    "get_gift_promotion": {"train": ["看看买这件送什么", "确认满额附带礼品是否还有", "查赠送规则和随单条件", "核对附带礼品能不能选"], "test": ["看这单有没有附赠", "确认赠送品是否还有余量"]},
    "compare_promotion_options": {"train": ["比较用券会员价积分哪种最便宜", "算不同促销组合后的最终金额", "在几套优惠搭配里选省钱方案", "对比直降和满减叠加效果"], "test": ["帮我选最划算的优惠搭配", "比较几种省钱方案的实付差异"]},
    "get_order_logistics": {"train": ["看看包裹现在走到哪里", "查快递当前节点和公司", "确认预计哪天送到", "追踪这单运输轨迹"], "test": ["帮我看包裹到哪了", "查这件货预计送达时间"]},
    "estimate_delivery_time": {"train": ["按我的地址估算几天能到", "看看这个仓发出大概要多久", "根据配送方式预估到达日", "判断今天下单什么时候送达"], "test": ["帮我估一下到货时间", "看寄到这个地址需要几天"]},
    "get_delivery_options": {"train": ["看看可选快递还是自取", "列出这单支持哪些送达方式", "确认有没有同城或安装配送", "查可预约的送货方案"], "test": ["帮我看有哪些收货选择", "确认能不能门店拿或约时间送"]},
    "select_delivery_slot": {"train": ["把送货时间选到周六上午", "修改可预约订单的送达时段", "给这单安排晚上配送", "选择合适的日期和时间窗口"], "test": ["帮我约一个送货时间段", "把配送改到明天下午"]},
    "get_pickup_points": {"train": ["找附近可以自取的网点", "看看周边取货点营业到几点", "查哪个站点能存放这个包裹", "确认取件需要什么材料"], "test": ["帮我找最近的自取位置", "查看附近网点可保存多久"]},
    "change_delivery_method": {"train": ["把快递改成到店拿", "看看这单能不能从自取换成送上门", "切换为同城即时送", "改用支持预约的送达方案"], "test": ["帮我换一种收货方式", "把这单改成门店取"]},
    "can_merge_ship": {"train": ["判断两单能不能一起发出", "看看新买的能否和旧单同包裹", "确认同地址同仓是否可合并寄送", "检查多件商品能否一起出库"], "test": ["帮我看能不能拼成一个包裹", "确认这些东西是否能一起发"]},
    "split_shipment_plan": {"train": ["看看为什么会分成多个包裹", "查询多件商品分别从哪里发", "确认预售和现货的拆开发送安排", "列出每个包裹包含哪些东西"], "test": ["帮我看这单怎么分开发", "确认几件商品各自的出库计划"]},
    "get_warehouse_stock_region": {"train": ["看看我所在省哪个仓有货", "查本地仓能不能直接发", "确认跨区调拨是否可行", "查看不同地区可售数量"], "test": ["帮我看附近仓是否有货", "查发货仓和调拨可能"]},
    "query_delivery_fee": {"train": ["算寄到这个地址要多少钱", "看看会员能否免运送费", "按重量和地区估算配送费用", "确认活动后还需不需要付邮费"], "test": ["帮我算这单邮寄费用", "看这个地址收货要不要运送费"]},
    "get_lost_package_process": {"train": ["包裹很久不动时该怎么处理", "看看疑似丢失能否赔付", "查物流停滞后的工单入口", "确认长时间未送达的处理流程"], "test": ["帮我处理运输中断很久的问题", "查包裹可能丢了该走什么流程"]},
    "schedule_installation": {"train": ["给家电约师傅上门", "安排家具送达后的装配时间", "看看安装服务覆盖不覆盖我家", "预约大件商品的上门服务"], "test": ["帮我约安装师傅", "确认这单能不能安排上门装好"]},
    "get_return_policy": {"train": ["看看这类商品能不能无理由退", "确认退换时运送费谁承担", "查特殊品类的退回限制", "了解签收后多久还能处理"], "test": ["帮我看这件能不能退换", "确认退回期限和费用规则"]},
    "create_refund_request": {"train": ["为这单提交退钱退货", "选择原因并发起退回流程", "帮我申请只退钱不寄回", "把不想要的商品走退货"], "test": ["帮我发起退回并返钱", "这件我要退掉请提交"]},
    "get_refund_status": {"train": ["看看退钱审核到哪一步", "查寄回后多久到账", "确认退款失败的原因", "跟进价差返还进度"], "test": ["帮我看退回的钱什么时候到", "查询售后退款当前节点"]},
    "create_exchange_request": {"train": ["把这件换成另一个规格", "提交换同款新颜色的请求", "选择取件方式并发起调换", "因尺码不合申请更换"], "test": ["帮我把商品换一个码", "这件有问题想换新"]},
    "get_repair_service_policy": {"train": ["看看故障维修怎么收费", "确认保内能不能免费修", "查寄修方式和预计费用", "了解哪些问题属于维修范围"], "test": ["帮我看坏了能不能修", "确认维修流程和费用"]},
    "create_repair_ticket": {"train": ["为故障商品登记维修", "创建寄修处理单并给地址", "预约检修并填写问题现象", "提交返厂处理请求"], "test": ["帮我报修这个商品", "给这件故障品开维修处理"]},
    "get_aftersale_ticket_status": {"train": ["看看售后处理到哪个节点", "查询换新或修理预计完成时间", "跟进补寄工单当前进度", "确认售后单下一步要做什么"], "test": ["帮我查售后处理进展", "查看维修换货单什么时候完成"]},
    "create_aftersale_ticket": {"train": ["商品破损帮我登记处理", "质量异常需要提交售后", "少件和物流问题开处理单", "把收到坏货的情况记录给平台"], "test": ["帮我为破损商品提交处理", "这次收货有问题请建单"]},
    "request_reship_missing_item": {"train": ["少发配件时申请补发", "漏掉的赠品帮我让商家再寄", "提交缺少零件的补发请求", "登记包裹里少东西的问题"], "test": ["帮我申请把缺的配件寄来", "这单漏发了请安排补发"]},
    "get_refund_amount_estimate": {"train": ["预估退一件能返钱多少", "算扣除运送费和优惠后的返还金额", "看看部分退货后能回多少", "根据权益使用情况估算退回金额"], "test": ["帮我算大概能退回多少钱", "确认退掉其中一件的返还数"]},
    "upload_aftersale_evidence": {"train": ["上传破损照片给售后", "把开箱视频提交到处理单", "补充检测报告作为凭证", "提交问题图片并确认受理"], "test": ["帮我把售后材料传上去", "补交照片视频给处理人员"]},
    "cancel_refund_request": {"train": ["撤回还没完成的退钱流程", "取消刚提交的退货请求", "看看终止售后后订单能否恢复", "把退款单先关掉"], "test": ["帮我取消这次退货", "撤掉未完成的返钱申请"]},
    "get_complaint_channels": {"train": ["看看可以在哪里投诉商家", "查物流投诉入口和材料", "确认平台申诉需要准备什么", "找监管申诉路径说明"], "test": ["找投诉入口", "查看投诉需要哪些材料"]},
    "escalate_to_human_service": {"train": ["自动售后解决不了转人工", "把订单问题升级给客服处理", "带上证据让真人介入", "复杂售后请专员继续跟进"], "test": ["转给售后专员", "这个问题让客服专员处理"]},
    "get_quality_issue_policy": {"train": ["看看质量争议怎么判定", "确认破损和性能故障如何赔付", "查疑似假货的处理规则", "了解收到瑕疵品的赔偿标准"], "test": ["看瑕疵争议怎么处理", "确认破损商品的判责规则"]},
    "get_user_profile": {"train": ["看看账号常买什么类型", "整理我的基础购物偏好", "读取常用地址和会员信息", "了解这个用户的消费画像"], "test": ["帮我概览账号偏好", "查看我的购物基础信息"]},
    "get_browsing_history": {"train": ["翻我最近看过的商品", "查看前几天停留过的页面", "找刚才浏览过的类目", "按时间列出访问过的商品"], "test": ["帮我找回刚看过的东西", "查看近期浏览足迹"]},
    "get_purchase_history": {"train": ["看看以前买过哪些品牌", "查常买商品和复购周期", "翻历史消费里的商品摘要", "找过去购买过的同类东西"], "test": ["帮我回顾以前买过什么", "查看曾经购买的品牌和商品"]},
    "get_favorite_items": {"train": ["看看我保存过哪些商品", "找心愿单里的降价关注", "查看标记喜欢的商品", "翻账号保存过的想买清单"], "test": ["打开心愿清单内容", "查我关注降价的商品"]},
    "get_user_addresses": {"train": ["看看账号里有哪些收件地点", "找默认收货地点和服务范围", "列出可用的收货人信息", "确认哪个地址能配送"], "test": ["帮我查看保存的地址", "查当前默认收件地点"]},
    "update_default_address": {"train": ["把公司设成常用收件地", "更新默认收货地点并校验", "将新住址设为以后优先使用", "修改账号首选配送地址"], "test": ["帮我换掉默认收件地址", "把这个新地址设为首选"]},
    "get_size_profile": {"train": ["看看我保存的身高体重", "查账号里的鞋码和常穿号码", "读取试穿反馈里的尺寸偏好", "确认尺码资料是否完整"], "test": ["帮我查看身体尺寸档案", "查我平时穿多大码"]},
    "update_size_profile": {"train": ["把最近试穿偏大写进去", "更新我的身高体重和鞋码", "记录这次尺码反馈供以后参考", "修改账号里的穿着偏好"], "test": ["帮我更新常穿号码", "把试穿结果保存到尺寸资料"]},
    "get_recommendation_feed": {"train": ["按我的偏好生成商品流", "根据浏览购买记录推荐可买款", "结合收藏给我推一些商品", "生成适合我的首页推荐"], "test": ["帮我刷一组个性商品", "按账号偏好推荐一些选择"]},
    "get_notification_preferences": {"train": ["看看我订阅了哪些提醒", "查降价和到货通知是否开启", "确认物流售后消息的接收设置", "读取账号提醒接收设置"], "test": ["看通知订阅设置", "查哪些提醒现在会发给我"]},
    "create_cart_item": {"train": ["把这个红色中码放进待买清单", "加入两件当前选择的规格", "将商品先放到购物篮", "把选好的版本添加进去"], "test": ["帮我把这件先放进车里", "添加当前规格准备一起结算"]},
    "get_cart_summary": {"train": ["看看待买清单里有哪些东西", "汇总购物篮金额和数量", "确认哪些商品失效不能结算", "查看当前待买清单可付款情况"], "test": ["汇总车里的商品", "看购物篮总共多少钱"]},
    "update_cart_quantity": {"train": ["把车里这件改成三件", "减少购物篮里某商品数量", "修改数量后看看金额变化", "调整待买商品件数并校验余量"], "test": ["帮我把车里数量改一下", "这件只买一件请更新"]},
    "remove_cart_item": {"train": ["把车里这件删掉", "清理不能买的失效商品", "从购物篮移除选中的东西", "删除不想结算的那一项"], "test": ["帮我从车里拿掉这件", "把失效商品清出去"]},
    "save_item_for_later": {"train": ["把车里这件先留到以后买", "移到稍后再考虑的清单", "暂时不结算但帮我保存", "把当前商品放到以后购买"], "test": ["帮我先搁置这件商品", "这件晚点买请保存起来"]},
    "get_checkout_preview": {"train": ["生成付款前的确认页面", "预览商品地址费用和风险提示", "看看提交前总价与优惠", "核对结算页里的所有信息"], "test": ["帮我看下单前预览", "确认付款前页面有没有问题"]},
    "submit_order": {"train": ["确认无误后创建这笔购买", "把结算信息提交生成编号", "下单并返回待付款结果", "按当前购物车生成新交易"], "test": ["帮我正式下这单", "提交购买并给我交易编号"]},
    "validate_checkout_items": {"train": ["检查车里商品能不能结算", "确认限购余量地址都没问题", "校验下单前库存和配送限制", "看看购物篮里哪些不能买"], "test": ["帮我检查能否顺利付款", "确认提交前商品都可售"]},
    "detect_abnormal_order": {"train": ["判断这笔交易有没有异常风险", "检查金额地址频次是否可疑", "识别售后行为导致的下单风险", "评估订单是否触发异常规则"], "test": ["帮我排查这单为什么异常", "看这笔购买有没有风险特征"]},
    "get_sensitive_category_rules": {"train": ["查酒类购买有什么限制", "看看跨境商品配送要求", "确认药品类目下单条件", "了解虚拟商品的购买规则"], "test": ["帮我看特殊类目限制", "确认这类商品能不能配送"]},
    "check_purchase_limit": {"train": ["看看这个商品每人最多买几件", "校验地区和年龄限制", "确认活动购买次数是否超了", "检查账号还能不能继续买"], "test": ["帮我查是否超过购买上限", "确认这次还能不能按规则购买"]},
    "check_account_risk": {"train": ["看看账号为什么被限制用券", "检查登录和付款安全状态", "确认售后频繁是否影响购买", "查账户当前有哪些限制原因"], "test": ["帮我排查账号风险状态", "查看账号被限制的原因"]},
    "create_manual_service_ticket": {"train": ["为复杂问题建客服专员工单", "把风控和售后材料交给专员", "创建需要客服处理的记录", "提交人工协助并附上订单信息"], "test": ["开一个人工处理单", "把这个复杂问题交给客服"]},
    "get_service_queue_status": {"train": ["看看人工客服还要排多久", "查询工单当前由谁处理", "确认预计什么时候响应", "查看客服队列里的进度"], "test": ["帮我看客服排到哪了", "查人工单预计多久回复"]},
    "generate_risk_notice": {"train": ["给用户生成购买限制说明", "把付款失败原因写成提示文案", "为特殊类目限制整理提醒", "输出异常交易的用户可读说明"], "test": ["帮我写一段风险提醒", "把不能购买的原因整理成提示"]},
}


def _render(seed: dict[str, list[str]]) -> dict[str, list[str]]:
    train = [template.format(text) for template, text in zip(_TRAIN_TEMPLATES, seed["train"])]
    test = [template.format(text) for template, text in zip(_TEST_TEMPLATES, seed["test"])]
    return {"train": train, "test": test}


def phrase_bank() -> dict[str, dict[str, list[str]]]:
    """Return train/test isolated expression seeds keyed by tool_id."""

    return {tool_id: _render(seed) for tool_id, seed in _SEEDS.items()}


def _load_catalog(path: Path = _CATALOG_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _catalog_triggers(tool: dict[str, Any]) -> list[str]:
    values = [tool.get("tool_id", ""), tool.get("name_zh", ""), tool.get("keyword_label", "")]
    values.extend(tool.get("positive_phrases") or [])
    return [str(value) for value in values if str(value)]


def validate_phrase_bank(catalog: list[dict]) -> list[str]:
    """Validate coverage, split isolation, and exact catalog trigger leakage."""

    errors: list[str] = []
    bank = phrase_bank()
    catalog_ids = {tool["tool_id"] for tool in catalog}
    bank_ids = set(bank)

    for tool_id in sorted(catalog_ids - bank_ids):
        errors.append(f"missing tool_id in phrase bank: {tool_id}")
    for tool_id in sorted(bank_ids - catalog_ids):
        errors.append(f"unknown tool_id in phrase bank: {tool_id}")

    for tool in catalog:
        tool_id = tool["tool_id"]
        if tool_id not in bank:
            continue
        item = bank[tool_id]
        train = item.get("train", [])
        test = item.get("test", [])
        if len(train) < 4:
            errors.append(f"{tool_id}: train has {len(train)} phrases, expected >= 4")
        if len(test) < 2:
            errors.append(f"{tool_id}: test has {len(test)} phrases, expected >= 2")
        if set(train) & set(test):
            errors.append(f"{tool_id}: train/test intersection: {sorted(set(train) & set(test))}")

        triggers = _catalog_triggers(tool)
        for split_name, phrases in (("train", train), ("test", test)):
            seen: set[str] = set()
            for phrase in phrases:
                if phrase in seen:
                    errors.append(f"{tool_id}: duplicate phrase in {split_name}: {phrase}")
                seen.add(phrase)
                for trigger in triggers:
                    if trigger in phrase:
                        errors.append(
                            f"{tool_id}: {split_name} phrase contains catalog trigger "
                            f"{trigger!r}: {phrase}"
                        )

    return errors


def main() -> None:
    errors = validate_phrase_bank(_load_catalog())
    print(f"errors: {len(errors)}")
    for error in errors[:20]:
        print(error)


if __name__ == "__main__":
    main()
