# learning batching data
for i in range(len(batch_y)):
        # extract the predict proba for batch_y
        pre_proba = pred_y[i][int(np.argmax(batch_y[i]))]
        # set the exit condition
        success_flag = False
        no_of_attempts = 0
        # retrain on the single input and output
        while pre_proba <= desired_proba and (no_of_attempts<attempts):
            print(pre_proba)            
            exec_model.fit(np.reshape(batch_x[i],(1,-1)), np.reshape(batch_y[i],(1,-1)))
            
            no_of_attempts += 1

            pred_one_y = exec_model.predict_proba(np.reshape(batch_x[i],(1,-1)), verbose=2)
            pre_proba = pred_one_y[0][int(np.argmax(batch_y[i]))]
            
            print("Attempt Number %d, Predicted Proba for this iteration %f" %(no_of_attempts, pre_proba))

            if pre_proba > desired_proba:
                success_flag = True
                break

        if (success_flag == False) and (no_of_attempts >= attempts):
            print("[-] Failed to incorporate this feedback")

        if success_flag == True:
            print("[+] Feedback incorporated \n")
            print("Took %d iterations to learn!" %(no_of_attempts))
