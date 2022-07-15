from flask import Flask, jsonify, render_template, redirect, request
import joblib
import js2py
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("new.html")


@app.route("/predict", methods=["GET", "POST"])
def result():
    print("helloworld")
    item_weight = float(request.form['item_weight'])
    item_fat_content = float(request.form['item_fat_content'])
    item_visibility = float(request.form['item_visibility'])
    item_type = float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year = float(request.form['outlet_establishment_year'])
    outlet_size = float(request.form['outlet_size'])
    outlet_location_type = float(request.form['outlet_location_type'])
    outlet_type = float(request.form['outlet_type'])

    X = np.array([[item_weight, item_fat_content, item_visibility, item_type, item_mrp,
                   outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]])

    scaler_path = r'C:\Users\Roshan\Desktop\Re_Prediction\models\sc.sav'

    sc = joblib.load(scaler_path)

    X_std = sc.transform(X)

    model_path = r'C:\Users\Roshan\Desktop\Re_Prediction\models\rf.sav'

    model = joblib.load(model_path)

    Y_pred = model.predict(X_std)


    if item_fat_content == 0:
        item_fat_content = 'High Fat'
    elif item_fat_content == 1:
        item_fat_content = 'Low Fat'
    else:
        item_fat_content = 'Regular'

    if item_type == 0:
        item_type = 'Baking Goods'
    elif item_type == 1:
        item_type = 'Breads'
    elif item_type == 2:
        item_type = 'Breakfasts'
    elif item_type == 3:
        item_type = 'Canned'
    elif item_type == 4:
        item_type = 'Diary'
    elif item_type == 5:
        item_type = 'Frozen Foods'
    elif item_type == 6:
        item_type = 'Fruits and Vegetables'
    elif item_type == 7:
        item_type = 'Hard Drinks'
    elif item_type == 8:
        item_type = 'Health and Hygiene'
    elif item_type == 9:
        item_type = 'Household'
    elif item_type == 10:
        item_type = 'Meat'
    elif item_type == 11:
        item_type = 'Others'
    elif item_type == 12:
        item_type = 'Seafood'
    elif item_type == 13:
        item_type = 'Snack Foods'
    elif item_type == 14:
        item_type = 'Soft Drinks'
    else:
        item_type = 'Starchy Foods'

    if outlet_size == 0:
        outlet_size = 'High'
    elif outlet_size == 1:
        outlet_size = 'Medium'
    else:
        outlet_size = 'Small'

    if outlet_location_type == 0:
        outlet_location_type = 'Tier-1'
    elif outlet_location_type == 1:
        outlet_location_type = 'Tier-2'
    else:
        outlet_location_type = 'Tier-3'

    if outlet_type == 0:
        outlet_type = 'Grocery Store'
    elif outlet_type == 1:
        outlet_type = 'Supermarket Type1'
    elif outlet_type == 1:
        outlet_type = 'Supermarket Type2'
    else:
        outlet_type = 'Supermarket Type3'


    return render_template("prediction.html", result=float(Y_pred), itemWeight=item_weight, itemFatContent=item_fat_content, itemVisibility=item_visibility, itemType=item_type, itemMrp=item_mrp, outletEstablishmentYear=outlet_establishment_year, outletSize=outlet_size, outletLocationType=outlet_location_type, outletType=outlet_type )

@app.route('/second')
def second():
    return render_template('new.html')


if __name__ == "__main__":
    app.run(debug=True, port=9457)
