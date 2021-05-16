import sys
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import json
import timeit

cred = credentials.Certificate('test-f4dbf-firebase-adminsdk-owx6b-7ca968edf1.json')
firebase_admin.initialize_app(cred, {
  'projectId': 'test-f4dbf',
})

db = firestore.client()

uid = 'efg'

doc_ref = db.collection(u'users').document(uid)
doc_ref.set(
    {
    u'first': u'asf',
    u'last': u'Lovelace',
    u'born': 1815
}
) # data write test
