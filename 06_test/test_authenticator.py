import pytest
from authenticator import Authenticator

@pytest.fixture
def authenticator():
    auth = Authenticator()
    yield auth

def test_register(authenticator):
    authenticator.register("yoshima", "hoge")
    assert authenticator.users == {"yoshima": "hoge"}

def test_register_error(authenticator):
    authenticator.register("yoshima", "hoge")    
    with pytest.raises(ValueError, match="エラー: ユーザーは既に存在します。"):
        authenticator.register("yoshima", "hoge")

def test_login(authenticator):
    authenticator.register("yoshima", "hoge")
    res = authenticator.login("yoshima", "hoge")
    assert res == "ログイン成功"

def test_login_error(authenticator):
    authenticator.register("yoshima", "hoge")
    with pytest.raises(ValueError, match="エラー: ユーザー名またはパスワードが正しくありません。"):
        authenticator.login("yoshima", "hogehoge")
